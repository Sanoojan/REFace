"""utilities for creating and reshaping grid points."""

import torch
from torchvision.transforms import ToTensor, ToPILImage


TOTEN = ToTensor()
TOPIL = ToPILImage()
DEVICE = torch.device("cpu")
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def grid_points_2d(width, height, device=DEVICE):
    """
    Create 2d grid points. Distribute points across a width and height,
    with the resulting coordinates constrained to -1, 1
    returns tensor shape (width * height, 2)
    """
    xx, yy = torch.meshgrid(
        [torch.linspace(-1.0, 1.0, height, device=device),
         torch.linspace(-1.0, 1.0, width, device=device)])
    return torch.stack([yy, xx], dim=-1).contiguous().view(-1, 2)


def noisy_grid(width, height, noise_scale=0.1, device=DEVICE):
    """
    Make uniform grid points, and add noise except for edge points.
    """
    grid = grid_points_2d(width, height, device)
    mod = torch.zeros([height, width, 2], device=device)
    noise = (torch.rand([height - 2, width - 2, 2]) - 0.5) * noise_scale
    mod[1:height - 1, 1:width-1, :] = noise
    return grid + mod.reshape(-1, 2)


def grid_to_img(grid_points, width, height):
    """
    convert (N * 2) tensor of grid points in -1, 1 to tuple of (x, y)
    scaled to width, height.
    return (x, y) to plot"""
    grid_clone = grid_points.clone().detach().cpu().numpy()
    x = (1 + grid_clone[..., 0]) * (width - 1) / 2
    y = (1 + grid_clone[..., 1]) * (height - 1) / 2
    return x.flatten(), y.flatten()
