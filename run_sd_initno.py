import os
import torch

from initno.pipelines.pipeline_sd_initno import StableDiffusionInitNOPipeline


# ---------
# Arguments
# ---------
SEEDS           = [0]
SD14_VERSION    = "CompVis/stable-diffusion-v1-4"
SD15_VERSION    = "runwayml/stable-diffusion-v1-5"
PROMPT          = "a cat and a rabbit"
token_indices   = [2, 5]
result_root     = "results"

os.makedirs('{:s}'.format(result_root), exist_ok=True)

def main():

    pipe = StableDiffusionInitNOPipeline.from_pretrained(SD14_VERSION).to("cuda")

    # use get_indices function to find out indices of the tokens you want to alter
    pipe.get_indices(PROMPT)

    for SEED in SEEDS:

        print('Seed ({}) Processing the ({}) prompt'.format(SEED, PROMPT))

        generator = torch.Generator("cuda").manual_seed(SEED)
        images = pipe(
            prompt=PROMPT,
            token_indices=token_indices,
            guidance_scale=7.5,
            generator=generator,
            num_inference_steps=50,
            max_iter_to_alter=25,
            result_root=result_root,
            seed=SEED,
            run_sd=False,
        ).images

        image = images[0]
        image.save(f"./{result_root}/{PROMPT}_{SEED}.jpg")


if __name__ == '__main__':
    main()