from torch import nn

from .BaseAgent import BaseAgent


class RamAgent(BaseAgent):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            #nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            #nn.BatchNorm1d(128),
            nn.Linear(128, 32),
            nn.ReLU(),
            #nn.BatchNorm1d(32),
            nn.Linear(32, output_size),
        )

