import torch
import numpy as np

from collections import defaultdict

from ...Device import device


class RollingState(torch.Tensor):
    @staticmethod
    def __new__(cls, time_depth, s, *args, **kwargs):
        if isinstance(s, (list, tuple)):
            s = torch.tensor(s)
        # s maybe time slice to initialize rolling state, or (s, *args)
        # may be shape to initialize emtpy (zero'd) rolling state
        if torch.is_tensor(s) or isinstance(s, np.ndarray):
            self = super().__new__(cls, time_depth+1, *s.shape, *args, **kwargs).to(device)
            self.fill_(.0)
            self.update(s)
        else:  # assume (s, *args) holds rest of shape  (shape=(time_depth+1, s, *args))
            self = super().__new__(cls, time_depth+1, s, *args, **kwargs).to(device)
            self.fill_(.0)

        return self

    def update(self, s: torch.Tensor) -> torch.Tensor:
        self[0, ...].copy_(s)
        self[...] = self.roll(-1, 0)[...]
        return self

    @property
    def now(self):
        return self[:-1]

    @property
    def next(self):
        return self[1:]

    def time_split(self):  # for s_t and s_{t+1}
        return self.now, self.next

    def copy(self):
        shape = self.shape
        new = RollingState(shape[0]-1, *shape[1:])
        new.copy_(self)
        return new


class ActionTracker:
    def __init__(self, action_map=None):
        self.reset()
        self.action_map = action_map

    def update(self, action):
        self.count += 1
        self.actions[action] += 1

    def reset(self):
        self.actions = defaultdict(lambda: 0)
        self.count = 0

    def __str__(self):
        C = self.count
        items = self.actions.items()
        items = sorted(items)
        if self.action_map:
            items = ((self.action_map[a], c) for a, c in items)
        return str(C) + ' ' + ' '.join(f"{a}:{c/C:0.2f}" for a,c in items)

