import torch

from torch import nn

from .ImageLayers import ImageEater
from .BaseAgent import BaseAgent


class VisualAgent(BaseAgent):
    def __init__(self, in_channels, output_size, size=84, **image_eater_args):
        super().__init__()
        self.image_eater = ImageEater(size=size, **image_eater_args)
        self.model = nn.Sequential(  #show_shape("input"),
            nn.Conv2d(in_channels, 64, 3), #show_shape("Conv2d a"),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=2), #show_shape("Conv2d b"),
            nn.ReLU(),
            nn.BatchNorm2d(32), #show_shape("BatchNorm2d"),
            nn.Conv2d(32, 1, 1), #show_shape("squeeze layer"),
            nn.Flatten(), #show_shape("Flatten"),
            nn.ReLU(),
            nn.Linear(((size-4)//2)**2, output_size),
        )

    def process_observation(self, obs):
        return self.image_eater(obs)

    def save(self, fname):
        eater = self.image_eater
        self.image_eater = str(eater)
        super().save(fname)
        self.image_eater = eater
        return self


class show_shape(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(self.name)
        print(x.shape)
        return x

