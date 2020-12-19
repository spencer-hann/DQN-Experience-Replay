import random

from collections import deque, namedtuple

from .Utils import RollingState


class ReplayMemory(deque):
    Experience = namedtuple("Experience", ('s', 'a', 'r', 'done'))

    def __init__(self, l=None, maxlen=None):
        if l is None:
            l = []  # maxlen stays None for deque.__init__
        elif isinstance(l, int):
            maxlen = l
            l = []

        super().__init__(l, maxlen)

    def store(
        self,
        s: RollingState,
        a: int,
        r: float,
        done: bool,
    ):
        if r or done or random.random() < .01:
        # 1% chance to store "boring" experience
            s = s.copy()
            expr = ReplayMemory.Experience(s, a, r, done)
            self.append(expr)
        return self

    def _sample(self, k):
        # random.sample without replacement
        return random.sample(self, k=k)

    def sample(self, k):
        if len(self) < k:
            # random.choices with replacement
            return random.choices(self, k=k,)
        self.sample = self._sample
        return self.sample(k)

    #def careful_sample(self, k, *args, with_replacement=False, **kwargs):
    #    if len(self) < k or with_replacement:
    #        # random.choices automatically samples w/ replacement
    #        return random.choices(self, k=k, *args, **kwargs)
    #    return random.sample(self, k=k, *args, **kwargs)

