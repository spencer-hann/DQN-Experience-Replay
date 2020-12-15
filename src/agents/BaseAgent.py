import torch


class BaseAgent(torch.nn.Module):
    def process_observation(self, obs):
        return torch.from_numpy(obs)

    @property
    def name(self):
        return type(self).__name__

    def save(self, fname):
        torch.save(self, fname)
        return self

