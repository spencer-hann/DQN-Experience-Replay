import torch

from torch import nn

from .ImageLayers import ImageEater


def VisualAgent(im_resize_factor, output_size):
    return nn.Sequential(
        ImageEater(im_resize_factor),
        nn.Conv2d(1, 64, 3),
        nn.ReLU(),
        nn.Conv2d(64, 32, 3),
        nn.Flatten(),
        nn.Relu(),
        nn.Linear(?, output_size),
    )

