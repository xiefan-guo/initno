import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import logging
import numpy as np
import torch
import torch.utils.checkpoint as checkpoint
from torch.nn import functional as F
from torch.optim.adam import Adam
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin, FromSingleFileMixin, IPAdapterMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from utils.ptp_utils import AttendExciteAttnProcessor, AttentionStore
from utils.attn_utils import fn_show_attention, fn_smoothing_func, fn_get_topk, fn_clean_mask, fn_get_otsu_mask
from tqdm import tqdm
import matplotlib.pyplot as plt


logging.basicConfig(format='%(asctime)s: %(message)s',level=logging.INFO)


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionAttendAndExcitePipeline

        >>> pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(
        ...     "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
        ... ).to("cuda")


        >>> prompt = "a cat and a frog"

        >>> # use get_indices function to find out indices of the tokens you want to alter
        >>> pipe.get_indices(prompt)
        {0: '<|startoftext|>', 1: 'a</w>', 2: 'cat</w>', 3: 'and</w>', 4: 'a</w>', 5: 'frog</w>', 6: '<|endoftext|>'}

        >>> token_indices = [2, 5]
        >>> seed = 6141
        >>> generator = torch.Generator("cuda").manual_seed(seed)

        >>> images = pipe(
        ...     prompt=prompt,
        ...     token_indices=token_indices,
        ...     guidance_scale=7.5,
        ...     generator=generator,
        ...     num_inference_steps=50,
        ...     max_iter_to_alter=25,
        ... ).images

        >>> image = images[0]
        >>> image.save(f"../images/{prompt}_{seed}.png")
        ```
"""


class StableDiffusionInitNOPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, IPAdapterMixin, FromSingleFileMixin):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion and Attend-and-Excite and Latent Consistency Models.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _exclude_from_cpu_offload = ["safety_checker"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        self.K = 1
        self.cross_attention_maps_cache = None

        if safety_checker is None and requires_safety_checker:
            logging.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logging.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        indices,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        indices_is_list_ints = isinstance(indices, list) and isinstance(indices[0], int)
        indices_is_list_list_ints = (
            isinstance(indices, list) and isinstance(indices[0], list) and isinstance(indices[0][0], int)
        )

        if not indices_is_list_ints and not indices_is_list_list_ints:
            raise TypeError("`indices` must be a list of ints or a list of a list of ints")

        if indices_is_list_ints:
            indices_batch_size = 1
        elif indices_is_list_list_ints:
            indices_batch_size = len(indices)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if indices_batch_size != prompt_batch_size:
            raise ValueError(
                f"indices batch size must be same as prompt batch size. indices batch size: {indices_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def fn_augmented_compute_losss(
        self, 
        indices: List[int], 
        smooth_attentions: bool = True,
        K: int = 1,
        attention_res: int = 16,) -> torch.Tensor:

        # -----------------------------
        # cross-attention response loss
        # -----------------------------
        aggregate_cross_attention_maps = self.attention_store.aggregate_attention(
            from_where=("up", "down", "mid"), is_cross=True)
        
        # cross attention map preprocessing
        cross_attention_maps = aggregate_cross_attention_maps[:, :, 1:-1]
        cross_attention_maps = cross_attention_maps * 100
        cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps, dim=-1)

        # Shift indices since we removed the first token
        indices = [index - 1 for index in indices]

        # clean_cross_attention_loss
        clean_cross_attn_loss = 0.

        # Extract the maximum values
        topk_value_list, topk_coord_list_list = [], []
        for i in indices:
            cross_attention_map_cur_token = cross_attention_maps[:, :, i]
            if smooth_attentions: cross_attention_map_cur_token = fn_smoothing_func(cross_attention_map_cur_token)
            topk_coord_list, _ = fn_get_topk(cross_attention_map_cur_token, K=K)

            topk_value = 0
            for coord_x, coord_y in topk_coord_list: topk_value = topk_value + cross_attention_map_cur_token[coord_x, coord_y]
            topk_value = topk_value / K

            topk_value_list.append(topk_value)
            topk_coord_list_list.append(topk_coord_list)

            # -----------------------------------
            # clean cross_attention_map_cur_token
            # -----------------------------------
            clean_cross_attention_map_cur_token                     = cross_attention_map_cur_token
            clean_cross_attention_map_cur_token_mask                = fn_get_otsu_mask(clean_cross_attention_map_cur_token)
            clean_cross_attention_map_cur_token_mask                = fn_clean_mask(clean_cross_attention_map_cur_token_mask, topk_coord_list[0][0], topk_coord_list[0][1])
            clean_cross_attention_map_cur_token_foreground          = clean_cross_attention_map_cur_token * clean_cross_attention_map_cur_token_mask + (1 - clean_cross_attention_map_cur_token_mask)
            clean_cross_attention_map_cur_token_background          = clean_cross_attention_map_cur_token * (1 - clean_cross_attention_map_cur_token_mask)

            if clean_cross_attention_map_cur_token_background.max() > clean_cross_attention_map_cur_token_foreground.min():
                clean_cross_attn_loss = clean_cross_attn_loss + clean_cross_attention_map_cur_token_background.max()
            else: clean_cross_attn_loss = clean_cross_attn_loss + clean_cross_attention_map_cur_token_background.max() * 0

        cross_attn_loss_list = [max(0 * curr_max, 1.0 - curr_max) for curr_max in topk_value_list]
        cross_attn_loss = max(cross_attn_loss_list)

        # ------------------------------
        # cross attention alignment loss
        # ------------------------------
        alpha = 0.9
        if self.cross_attention_maps_cache is None: self.cross_attention_maps_cache = cross_attention_maps.detach().clone()
        else: self.cross_attention_maps_cache = self.cross_attention_maps_cache * alpha + cross_attention_maps.detach().clone() * (1 - alpha)

        cross_attn_alignment_loss = 0
        for i in indices:
            cross_attention_map_cur_token = cross_attention_maps[:, :, i]
            if smooth_attentions: cross_attention_map_cur_token = fn_smoothing_func(cross_attention_map_cur_token)
            cross_attention_map_cur_token_cache = self.cross_attention_maps_cache[:, :, i]
            if smooth_attentions: cross_attention_map_cur_token_cache = fn_smoothing_func(cross_attention_map_cur_token_cache)
            cross_attn_alignment_loss = cross_attn_alignment_loss + torch.nn.L1Loss()(cross_attention_map_cur_token, cross_attention_map_cur_token_cache)          

        # ----------------------------
        # self-attention conflict loss
        # ----------------------------
        self_attention_maps = self.attention_store.aggregate_attention(
            from_where=("up", "down", "mid"), is_cross=False)

        self_attention_map_list = []
        for topk_coord_list in topk_coord_list_list:
            self_attention_map_cur_token_list = []
            for coord_x, coord_y in topk_coord_list:

                self_attention_map_cur_token = self_attention_maps[coord_x, coord_y]
                self_attention_map_cur_token = self_attention_map_cur_token.view(attention_res, attention_res).contiguous()
                self_attention_map_cur_token_list.append(self_attention_map_cur_token)

            if len(self_attention_map_cur_token_list) > 0:
                self_attention_map_cur_token = sum(self_attention_map_cur_token_list) / len(self_attention_map_cur_token_list)
                if smooth_attentions: self_attention_map_cur_token = fn_smoothing_func(self_attention_map_cur_token)
            else:
                self_attention_map_per_token = torch.zeros_like(self_attention_maps[0, 0])
                self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()

            self_attention_map_list.append(self_attention_map_cur_token)

        self_attn_loss, number_self_attn_loss_pair = 0, 0
        number_token = len(self_attention_map_list)
        for i in range(number_token):
            for j in range(i + 1, number_token): 
                number_self_attn_loss_pair = number_self_attn_loss_pair + 1
                self_attention_map_1 = self_attention_map_list[i]
                self_attention_map_2 = self_attention_map_list[j]

                self_attention_map_min = torch.min(self_attention_map_1, self_attention_map_2) 
                self_attention_map_sum = (self_attention_map_1 + self_attention_map_2)
                cur_self_attn_loss = (self_attention_map_min.sum() / (self_attention_map_sum.sum() + 1e-6))
                self_attn_loss = self_attn_loss + cur_self_attn_loss

        if number_self_attn_loss_pair > 0: self_attn_loss = self_attn_loss / number_self_attn_loss_pair

        joint_loss = cross_attn_loss * 1. + clean_cross_attn_loss * 0.1 + cross_attn_alignment_loss * 0.1 + self_attn_loss * 1.
        return joint_loss, cross_attn_loss, clean_cross_attn_loss, self_attn_loss
    
    def fn_calc_kld_loss_func(self, log_var, mu):
        return torch.mean(-0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp()), dim=0)
    
    def fn_compute_loss(
        self, 
        indices: List[int], 
        smooth_attentions: bool = True,
        K: int = 1,
        attention_res: int = 16,) -> torch.Tensor:
        
        # -----------------------------
        # cross-attention response loss
        # -----------------------------
        aggregate_cross_attention_maps = self.attention_store.aggregate_attention(
            from_where=("up", "down", "mid"), is_cross=True)
        
        # cross attention map preprocessing
        cross_attention_maps = aggregate_cross_attention_maps[:, :, 1:-1]
        cross_attention_maps = cross_attention_maps * 100
        cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps, dim=-1)

        # Shift indices since we removed the first token
        indices = [index - 1 for index in indices]

        # clean_cross_attention_loss
        clean_cross_attention_loss = 0.

        # Extract the maximum values
        topk_value_list, topk_coord_list_list = [], []
        for i in indices:
            cross_attention_map_cur_token = cross_attention_maps[:, :, i]
            if smooth_attentions: cross_attention_map_cur_token = fn_smoothing_func(cross_attention_map_cur_token)
            
            topk_coord_list, _ = fn_get_topk(cross_attention_map_cur_token, K=K)

            topk_value = 0
            for coord_x, coord_y in topk_coord_list: topk_value = topk_value + cross_attention_map_cur_token[coord_x, coord_y]
            topk_value = topk_value / K

            topk_value_list.append(topk_value)
            topk_coord_list_list.append(topk_coord_list)

            # -----------------------------------
            # clean cross_attention_map_cur_token
            # -----------------------------------
            clean_cross_attention_map_cur_token                     = cross_attention_map_cur_token
            clean_cross_attention_map_cur_token_mask                = fn_get_otsu_mask(clean_cross_attention_map_cur_token)
            clean_cross_attention_map_cur_token_mask                = fn_clean_mask(clean_cross_attention_map_cur_token_mask, topk_coord_list[0][0], topk_coord_list[0][1])
            
            clean_cross_attention_map_cur_token_foreground          = clean_cross_attention_map_cur_token * clean_cross_attention_map_cur_token_mask + (1 - clean_cross_attention_map_cur_token_mask)
            clean_cross_attention_map_cur_token_background          = clean_cross_attention_map_cur_token * (1 - clean_cross_attention_map_cur_token_mask)

            if clean_cross_attention_map_cur_token_background.max() > clean_cross_attention_map_cur_token_foreground.min():
                clean_cross_attention_loss = clean_cross_attention_loss + clean_cross_attention_map_cur_token_background.max()
            else: clean_cross_attention_loss = clean_cross_attention_loss + clean_cross_attention_map_cur_token_background.max() * 0

        cross_attn_loss_list = [max(0 * curr_max, 1.0 - curr_max) for curr_max in topk_value_list]
        cross_attn_loss = max(cross_attn_loss_list)

        # ----------------------------
        # self-attention conflict loss
        # ----------------------------
        self_attention_maps = self.attention_store.aggregate_attention(
            from_where=("up", "down", "mid"), is_cross=False)
        
        self_attention_map_list = []
        for topk_coord_list in topk_coord_list_list:
            self_attention_map_cur_token_list = []
            for coord_x, coord_y in topk_coord_list:

                self_attention_map_cur_token = self_attention_maps[coord_x, coord_y]
                self_attention_map_cur_token = self_attention_map_cur_token.view(attention_res, attention_res).contiguous()
                self_attention_map_cur_token_list.append(self_attention_map_cur_token)

            if len(self_attention_map_cur_token_list) > 0:
                self_attention_map_cur_token = sum(self_attention_map_cur_token_list) / len(self_attention_map_cur_token_list)
                if smooth_attentions: self_attention_map_cur_token = fn_smoothing_func(self_attention_map_cur_token)
            else:
                self_attention_map_per_token = torch.zeros_like(self_attention_maps[0, 0])
                self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()

            self_attention_map_list.append(self_attention_map_cur_token)

        self_attn_loss, number_self_attn_loss_pair = 0, 0
        number_token = len(self_attention_map_list)
        for i in range(number_token):
            for j in range(i + 1, number_token): 
                number_self_attn_loss_pair = number_self_attn_loss_pair + 1
                self_attention_map_1 = self_attention_map_list[i]
                self_attention_map_2 = self_attention_map_list[j]

                self_attention_map_min = torch.min(self_attention_map_1, self_attention_map_2) 
                self_attention_map_sum = (self_attention_map_1 + self_attention_map_2) 
                cur_self_attn_loss = (self_attention_map_min.sum() / (self_attention_map_sum.sum() + 1e-6))
                self_attn_loss = self_attn_loss + cur_self_attn_loss

        if number_self_attn_loss_pair > 0: self_attn_loss = self_attn_loss / number_self_attn_loss_pair

        cross_attn_loss = cross_attn_loss * torch.ones(1).to(self._execution_device)
        self_attn_loss  = self_attn_loss * torch.ones(1).to(self._execution_device)

        if cross_attn_loss > 0.5:    self_attn_loss = self_attn_loss * 0
        joint_loss = cross_attn_loss * 1. +  self_attn_loss * 1. + clean_cross_attention_loss * 1.

        return joint_loss, cross_attn_loss, self_attn_loss

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents

    def _perform_iterative_refinement_step(
        self,
        latents: torch.Tensor,
        indices: List[int],
        cross_attn_loss: torch.Tensor,
        self_attn_loss: torch.Tensor,
        clean_loss: torch.Tensor,
        threshold: float,
        text_embeddings: torch.Tensor,
        step_size: float,
        t: int,
        max_refinement_steps: int = 20,
    ):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent code
        according to our loss objective until the given threshold is reached for all tokens.
        """
        iteration = 0
        target_loss = max(0, 1.0 - threshold)
        target_self_loss = 0.3
        while cross_attn_loss > target_loss or self_attn_loss > target_self_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
            self.unet.zero_grad()

            joint_loss, cross_attn_loss, clean_loss, self_attn_loss = self.fn_augmented_compute_losss(indices=indices, K=1)
            if joint_loss != 0: latents = self._update_latent(latents, joint_loss, step_size)

            logging.info(f"\t Try {iteration}. cross loss: {cross_attn_loss:0.4f}. self loss: {self_attn_loss:0.4f}. clean loss: {clean_loss:0.4f}")

            if iteration >= max_refinement_steps:
                logging.info(f"\t Exceeded max number of iterations ({max_refinement_steps})! ")
                break

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        _ = self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
        self.unet.zero_grad()
        
        joint_loss, cross_attn_loss, clean_loss, self_attn_loss = self.fn_augmented_compute_losss(indices=indices, K=1)
        logging.info(f"\t Finished with loss of: {cross_attn_loss:0.4f}")
        return joint_loss, cross_attn_loss, self_attn_loss, clean_loss, latents, None

    # InitNO: Boosting Text-to-Image Diffusion Models via Initial Noise Optimization
    def fn_initno(
        self,
        latents: torch.Tensor,
        indices: List[int],
        text_embeddings: torch.Tensor,

        use_grad_checkpoint: bool = False,
        initno_lr: float = 1e-2,
        max_step: int = 50,
        round: int = 0,
        tau_cross_attn: float = 0.2,
        tau_self_attn: float = 0.3,
        num_inference_steps: int = 50,
        device: str = "",
        denoising_step_for_loss: int = 1,
        guidance_scale: int = 0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        do_classifier_free_guidance: bool = False,
    ):
        '''InitNO: Boosting Text-to-Image Diffusion Models via Initial Noise Optimization'''

        latents = latents.clone().detach()
        log_var, mu = torch.zeros_like(latents), torch.zeros_like(latents)
        log_var, mu = log_var.clone().detach().requires_grad_(True), mu.clone().detach().requires_grad_(True)
        optimizer = Adam([log_var, mu], lr=initno_lr, eps=1e-3)

        # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        optimization_succeed = False
        for iteration in tqdm(range(max_step)):

            optimized_latents = latents * (torch.exp(0.5 * log_var)) + mu
            
            # prepare scheduler
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # loss records
            joint_loss_list, cross_attn_loss_list, self_attn_loss_list = [], [], []
            
            # denoising loop
            for i, t in enumerate(timesteps):
                if i >= denoising_step_for_loss: break

                # Forward pass of denoising with text conditioning
                if use_grad_checkpoint:
                    noise_pred_text = checkpoint.checkpoint(self.unet, optimized_latents, t, text_embeddings[1].unsqueeze(0), use_reentrant=False).sample
                else: noise_pred_text = self.unet(optimized_latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample

                joint_loss, cross_attn_loss, self_attn_loss = self.fn_compute_loss(
                    indices=indices, K=1)
                joint_loss_list.append(joint_loss), cross_attn_loss_list.append(cross_attn_loss), self_attn_loss_list.append(self_attn_loss)

                if denoising_step_for_loss > 1:
                    with torch.no_grad():
                        if use_grad_checkpoint:
                            noise_pred_uncond = checkpoint.checkpoint(self.unet, optimized_latents, t, text_embeddings[0].unsqueeze(0), use_reentrant=False).sample
                        else: noise_pred_uncond = self.unet(optimized_latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample

                    if do_classifier_free_guidance: noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    optimized_latents = self.scheduler.step(noise_pred, t, optimized_latents, **extra_step_kwargs).prev_sample
                
            joint_loss      = sum(joint_loss_list) / denoising_step_for_loss
            cross_attn_loss = max(cross_attn_loss_list)
            self_attn_loss  = max(self_attn_loss_list)
            
            # print loss records
            joint_loss_list         = [_.item() for _ in joint_loss_list]
            cross_attn_loss_list    = [_.item() for _ in cross_attn_loss_list]
            self_attn_loss_list     = [_.item() for _ in self_attn_loss_list]

            if cross_attn_loss < tau_cross_attn and self_attn_loss < tau_self_attn:
                optimization_succeed = True
                break
  
            self.unet.zero_grad()
            optimizer.zero_grad()
            joint_loss = joint_loss.mean()
            joint_loss.backward()
            optimizer.step()

            # update kld_loss = self.fn_calc_kld_loss_func(log_var, mu)
            kld_loss = self.fn_calc_kld_loss_func(log_var, mu)
            while kld_loss > 0.001:
                optimizer.zero_grad()
                kld_loss = kld_loss.mean()
                kld_loss.backward()
                optimizer.step()
                kld_loss = self.fn_calc_kld_loss_func(log_var, mu)

        optimized_latents = (latents * (torch.exp(0.5 * log_var)) + mu).clone().detach()
        if self_attn_loss <= 1e-6: self_attn_loss = self_attn_loss + 1.
        return optimized_latents, optimization_succeed, cross_attn_loss + self_attn_loss 

    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttendExciteAttnProcessor(attnstore=self.attention_store, place_in_unet=place_in_unet)

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    def get_indices(self, prompt: str) -> Dict[str, int]:
        """Utility function to list the indices of the tokens you wish to alte"""
        ids = self.tokenizer(prompt).input_ids
        indices = {i: tok for tok, i in zip(self.tokenizer.convert_ids_to_tokens(ids), range(len(ids)))}
        return indices

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        token_indices: Union[List[int], List[List[int]]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_iter_to_alter: int = 25,
        scale_factor: int = 20,
        attn_res: Optional[Tuple[int]] = (16, 16),
        clip_skip: Optional[int] = None,
        
        result_root: str = '',
        seed: int = 0,
        K: int = 1,
        run_sd: bool = True,
        run_initno: bool = True
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            token_indices (`List[int]`):
                The token indices to alter with attend-and-excite.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            max_iter_to_alter (`int`, *optional*, defaults to `25`):
                Number of denoising steps to apply attend-and-excite. The `max_iter_to_alter` denoising steps are when
                attend-and-excite is applied. For example, if `max_iter_to_alter` is `25` and there are a total of `30`
                denoising steps, the first `25` denoising steps applies attend-and-excite and the last `5` will not.
            thresholds (`dict`, *optional*, defaults to `{0: 0.05, 10: 0.5, 20: 0.8}`):
                Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in.
            scale_factor (`int`, *optional*, default to 20):
                Scale factor to control the step size of each attend-and-excite update.
            attn_res (`tuple`, *optional*, default computed from width and height):
                The 2D resolution of the semantic attention map.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        self.cross_attention_maps_cache = None

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            token_indices,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clip_skip=clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if attn_res is None:
            attn_res = int(np.ceil(width / 32)), int(np.ceil(height / 32))
        self.attention_store = AttentionStore(attn_res)
        self.register_attention_control()

        # default config for step size from original repo
        scale_range = np.linspace(1.0, 0.5, len(self.scheduler.timesteps))
        step_size = scale_factor * np.sqrt(scale_range)

        text_embeddings = (
            prompt_embeds[batch_size * num_images_per_prompt :] if do_classifier_free_guidance else prompt_embeds
        )

        if isinstance(token_indices[0], int):
            token_indices = [token_indices]

        indices = []

        for ind in token_indices:
            indices = indices + [ind] * num_images_per_prompt

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # 8. initno
        run_initno = True
        if run_initno:
            max_round = 5
            with torch.enable_grad():
                optimized_latents_pool = []
                for round in range(max_round):
                    optimized_latents, optimization_succeed, cross_self_attn_loss = self.fn_initno(
                        latents=latents,
                        indices=token_indices[0],
                        text_embeddings=prompt_embeds,
                        max_step=10,
                        num_inference_steps=num_inference_steps,
                        device=device,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        eta=eta,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        round=round,
                    )
                    optimized_latents_pool.append((cross_self_attn_loss, round, optimized_latents.clone(), latents.clone(), optimization_succeed))
                    if optimization_succeed: break
            
                    latents = self.prepare_latents(
                        batch_size * num_images_per_prompt,
                        num_channels_latents,
                        height,
                        width,
                        prompt_embeds.dtype,
                        device,
                        generator,
                        latents=None,
                    )
                optimized_latents_pool.sort()
                # for score, _round, _optimized_latent, _latent, _optimization_succeed in optimized_latents_pool: 
                #     print(f'Optimization_succeed: {_optimization_succeed} - Attn score: {score.item():0.4f} - Round: {_round}')
                
                if optimized_latents_pool[0][4] is True:
                    latents = optimized_latents_pool[0][2]
                else:
                    optimized_latents, optimization_succeed, cross_self_attn_loss = self.fn_initno(
                        latents=optimized_latents_pool[0][3],
                        indices=token_indices[0],
                        text_embeddings=prompt_embeds,
                        max_step=50,
                        num_inference_steps=num_inference_steps,
                        device=device,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        eta=eta,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        round=round,
                    ) 
                    latents = optimized_latents
                

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # store attention map
        cross_attention_map_numpy_list, self_attention_map_numpy_list = [], []
        with self.progress_bar(total=num_inference_steps) as progress_bar:

            for i, t in enumerate(timesteps):
                
                # Attend and excite process
                with torch.enable_grad():
                    latents = latents.clone().detach().requires_grad_(True)
                    updated_latents = []
                    for latent, index, text_embedding in zip(latents, indices, text_embeddings):
                        # Forward pass of denoising with text conditioning
                        latent = latent.unsqueeze(0)
                        text_embedding = text_embedding.unsqueeze(0)

                        self.unet(
                            latent,
                            t,
                            encoder_hidden_states=text_embedding,
                            cross_attention_kwargs=cross_attention_kwargs,
                        ).sample
                        self.unet.zero_grad()

                        joint_loss, cross_attn_loss, clean_loss, self_attn_loss = self.fn_augmented_compute_losss(indices=index, K=1)

                        if result_root is not None:
                            
                            cross_attention_maps = self.attention_store.aggregate_attention(
                                from_where=("up", "down", "mid"), is_cross=True)
                            self_attention_maps = self.attention_store.aggregate_attention(
                                from_where=("up", "down", "mid"), is_cross=False)
                            
                            cross_attention_map_numpy, self_attention_map_numpy = fn_show_attention(
                                cross_attention_maps=cross_attention_maps,
                                self_attention_maps=self_attention_maps,
                                indices=index,
                                K=1,
                                attention_res=16,
                                smooth_attentions=True)

                            cross_attention_map_numpy_list.append(cross_attention_map_numpy)
                            self_attention_map_numpy_list.append(self_attention_map_numpy)

                        # If this is an iterative refinement step, verify we have reached the desired threshold for all
                        if i < max_iter_to_alter and (i == 10 or i == 20) and (cross_attn_loss > 0.2 or self_attn_loss > 0.3) and not run_sd and True:
                        
                            joint_loss, cross_attn_loss, self_attn_loss, clean_loss, latent, max_attention_per_index = self._perform_iterative_refinement_step(
                                latents=latent,
                                indices=index,
                                cross_attn_loss=cross_attn_loss,
                                self_attn_loss=self_attn_loss,
                                clean_loss=clean_loss,
                                threshold=0.8,
                                text_embeddings=text_embedding,
                                step_size=step_size[i],
                                t=t,
                            )

                        # Perform gradient update
                        if i < max_iter_to_alter and not run_sd:
                            if cross_attn_loss != 0 and True:
                                latent = self._update_latent(
                                    latents=latent,
                                    loss=cross_attn_loss,
                                    step_size=step_size[i],
                                )
                            # logging.info(f"Iteration {i:02d} - cross loss: {cross_attn_loss:0.4f} - self loss: {self_attn_loss:0.4f}")

                        updated_latents.append(latent)

                    latents = torch.cat(updated_latents, dim=0)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

            # show attention map at each timestep
            if result_root is not None:
                cross_attention_map_numpy       = np.concatenate(cross_attention_map_numpy_list, axis=-1)
                self_attention_map_numpy        = np.concatenate(self_attention_map_numpy_list, axis=-1)

                attention_map_numpy = np.concatenate((cross_attention_map_numpy, self_attention_map_numpy), axis=0)

                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.imshow(attention_map_numpy, cmap='YlOrRd')
                plt.savefig(f"./{result_root}/{prompt}_{seed}_attn.jpg", dpi=600, bbox_inches='tight', pad_inches=0)
                plt.close()

        # 8. Post-processing
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
