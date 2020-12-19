import torch

from torch import nn
from matplotlib import pyplot as plt

from .ImageLayers import ImageEater
from .BaseAgent import BaseAgent


class VisualAgent(BaseAgent):
    def __init__(self, in_channels, output_size, size=84, **image_eater_args):
        super().__init__()
        # -4 for 3x3 kernels,no padding; /2 stride=2; **2 for flatten
        #final_layer_size = 16 * (((size - 7) // 2)-2)**2
        final_layer_size = 16 * (((size - 7) // 4)-2)**2

        self.process_observation = ImageEater(size=size, **image_eater_args)

        size = (size - 7) // 1 + 1  # conv 0
        size = (size - 3) // 2 + 1  # conv 1
        size = (size - 3) // 2 + 1  # conv 2
        size **= 2  # 2d square image
        size *= 16  # n channels after conv 2
        final_layer_size = size

        self.render = RenderState()
        self.render.off()
        self.model = nn.Sequential(  #PrintShape("input"),
            self.render,
            nn.Conv2d(in_channels, 32, 7), #PrintShape("Conv2d a"),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), #PrintShape("Conv2d b"),
            nn.ReLU(),
            #nn.BatchNorm2d(32), #PrintShape("BatchNorm2d"),
            nn.Conv2d(32, 16, 3, stride=2), #PrintShape("squeeze layer"),
            nn.Flatten(), #PrintShape("Flatten"),
            nn.ReLU(),
            nn.Linear(final_layer_size, output_size),
        )

    def save(self, fname):
        im_eater = self.process_observation
        self.process_observation = str(im_eater)
        super().save(fname)
        self.process_observation = im_eater
        return self


class PrintShape(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(x.shape, '\t', self.name)
        return x


class RenderState(nn.Module):
    def __init__(self):
        super().__init__()
        self.on()
        self.close = plt.close

    def render(self, x):
        plt.close()
        plt.pause(.1)
        print()
        for i, t in enumerate(x.cpu().numpy()[0,::-1]):
            print(i)
            plt.imshow(t, cmap="gray")
            plt.show();
        return x

    def no_render(self, x):
        return x

    def on(self, on=True):
        if on:
            self.forward = self.render
        else:
            self.off()

    def off(self):
        self.forward = self.no_render

