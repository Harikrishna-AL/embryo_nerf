# import pygame
# from pygame.locals import *
# from OpenGL.GL import *
# from OpenGL.GLU import *
import numpy as np
from PIL import Image
import torch
import torch.nn as nn


def tor_to_meshgrid(tensor1, tensor2):
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)


def cumprod(tensor):
    cumprod = torch.cumprod(tensor, -1)
    cumprod = torch.roll(cumprod, 1, -1)
    cumprod[..., 0] = 1
    return cumprod


def compute_rays(height, width, focal_length, cam2world):
    x, y = tor_to_meshgrid(
        torch.arange(width).to(cam2world), torch.arange(height).to(cam2world)
    )
    directions = torch.stack(
        [
            (x - width * 0.5) / focal_length,
            (y - height * 0.5) / focal_length,
            torch.ones_like(x),
        ],
        dim=-1,
    )
    directions = directions[..., None, :]
    cam2world = cam2world[:, :, :3, :3].squeeze()
    ray_directions = torch.sum(directions * cam2world, dim=-1)
    ray_origins = cam2world[:3, -1].expand(ray_directions.shape)
    return ray_directions, ray_origins


def compute_query_points(
    ray_directions, ray_origins, near, far, num_samples, random=True
):
    depth_values = torch.linspace(near, far, num_samples).to(ray_origins)
    if random:
        shape = list(ray_origins.shape[:-1]) + [num_samples]
        depth_values = (
            depth_values
            + torch.rand(shape).to(ray_origins) * (far - near) / num_samples
        )
    query_points = (
        ray_origins[..., None, :]
        + ray_directions[..., None, :] * depth_values[..., :, None]
    )
    return query_points, depth_values


def render_volume(radiance_field, ray_origins, depth_values):
    sigma_a = nn.functional.relu(radiance_field[..., 3])
    rgb = torch.sigmoid(radiance_field[..., :3])
    one_e_10 = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device)
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod(1.0 - alpha + 1e-10)
    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * depth_values).sum(dim=-1)
    acc__map = weights.sum(-1)
    return rgb_map, depth_map, acc__map


def pos_encoding(tensor, num_encoding_functions=6, include_input=True, log_sample=True):
    encoding = [tensor] if include_input else []
    if log_sample:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0**0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def get_batches(points, chunksize=1024 * 8):
    return [points[i : i + chunksize] for i in range(0, points.shape[0], chunksize)]
