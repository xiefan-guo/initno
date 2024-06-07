import numpy as np
import torch
from torch.nn import functional as F

from utils.gaussian_smoothing import GaussianSmoothing


def fn_get_topk(attention_map, K=1):
    H, W = attention_map.size()
    attention_map_detach = attention_map.detach().view(H * W)
    topk_value, topk_index = attention_map_detach.topk(K, dim=0, largest=True, sorted=True)
    topk_coord_list = []

    for index in topk_index:
        index = index.cpu().numpy()
        coord = index // W, index % W
        topk_coord_list.append(coord)
    return topk_coord_list, topk_value


def fn_smoothing_func(attention_map):
    smoothing = GaussianSmoothing().to(attention_map.device)
    attention_map = F.pad(attention_map.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
    attention_map = smoothing(attention_map).squeeze(0).squeeze(0)
    return attention_map


def fn_show_attention(
    cross_attention_maps,
    self_attention_maps,
    indices,
    K=1,
    attention_res=16,
    smooth_attentions=True,
):

    cross_attention_map_list, self_attention_map_list = [], []

    # cross attention map preprocessing
    cross_attention_maps = cross_attention_maps[:, :, 1:-1]
    cross_attention_maps = cross_attention_maps * 100
    cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps, dim=-1)

    # Shift indices since we removed the first token
    indices = [index - 1 for index in indices]

    for i in indices:
        cross_attention_map_per_token = cross_attention_maps[:, :, i]
        if smooth_attentions: cross_attention_map_per_token = fn_smoothing_func(cross_attention_map_per_token)
        cross_attention_map_list.append(cross_attention_map_per_token)

    for i in indices:
        cross_attention_map_per_token = cross_attention_maps[:, :, i]
        topk_coord_list, topk_value = fn_get_topk(cross_attention_map_per_token, K=K)

        self_attention_map_per_token_list = []
        for coord_x, coord_y in topk_coord_list:

            self_attention_map_per_token = self_attention_maps[coord_x, coord_y]
            self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()
            self_attention_map_per_token_list.append(self_attention_map_per_token)

        if len(self_attention_map_per_token_list) > 0:
            self_attention_map_per_token = sum(self_attention_map_per_token_list) / len(self_attention_map_per_token_list)
            if smooth_attentions: self_attention_map_per_token = fn_smoothing_func(self_attention_map_per_token)
        else:
            self_attention_map_per_token = torch.zeros_like(self_attention_maps[0, 0])
            self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()

        norm_self_attention_map_per_token = (self_attention_map_per_token - self_attention_map_per_token.min()) / \
            (self_attention_map_per_token.max() - self_attention_map_per_token.min() + 1e-6)
        
        self_attention_map_list.append(norm_self_attention_map_per_token)

    # tensor to numpy
    cross_attention_map_numpy       = torch.cat(cross_attention_map_list, dim=0).cpu().detach().numpy()
    self_attention_map_numpy        = torch.cat(self_attention_map_list, dim=0).cpu().detach().numpy()

    return cross_attention_map_numpy, self_attention_map_numpy


import cv2

def fn_get_otsu_mask(x):

    x_numpy = x
    x_numpy = x_numpy.cpu().detach().numpy()
    x_numpy = x_numpy * 255
    x_numpy = x_numpy.astype(np.uint16)

    opencv_threshold, _ = cv2.threshold(x_numpy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    opencv_threshold = opencv_threshold * 1. / 255.

    otsu_mask = torch.where(
        x < opencv_threshold,
        torch.tensor(0, dtype=x.dtype, device=x.device),
        torch.tensor(1, dtype=x.dtype, device=x.device))
    
    return otsu_mask


def fn_clean_mask(otsu_mask, x, y):
    
    H, W = otsu_mask.size()
    direction = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    def dfs(cur_x, cur_y):
        if cur_x >= 0 and cur_x < H and cur_y >= 0 and cur_y < W and otsu_mask[cur_x, cur_y] == 1:
            otsu_mask[cur_x, cur_y] = 2
            for delta_x, delta_y in direction:
                dfs(cur_x + delta_x, cur_y + delta_y)
    
    dfs(x, y)
    ret_otsu_mask = torch.where(
        otsu_mask < 2,
        torch.tensor(0, dtype=otsu_mask.dtype, device=otsu_mask.device),
        torch.tensor(1, dtype=otsu_mask.dtype, device=otsu_mask.device))

    return ret_otsu_mask