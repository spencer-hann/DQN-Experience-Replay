from torch import nn


def RamAgent(input_size, output_size):
    agent = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, output_size),
    )
    return agent

