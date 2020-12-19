import torch

from torch import tensor
from random import random


class BaseAgent(torch.nn.Module):
    def __init__(self, epsilon = 0.05):
        super().__init__()
        self.process_observation = torch.from_numpy
        self.epsilon = epsilon

    def forward(self, x):
        return self.model(x)

    @property
    def name(self):
        return type(self).__name__

    def save(self, fname):
        torch.save(self, fname)
        return self

    def select_action(self, state, action_space, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        if random() < epsilon:
            a = action_space.sample()
            return tensor(a), tensor(0.0)

        with torch.no_grad():
            q, a = self(state).max(dim=1)

        return a, q

