import torch
import random
import numpy as np

from torch import tensor
from collections import deque, namedtuple, defaultdict
from time import sleep

from .Schedulers import Repeater
from ..Device import device


VERBOSE = 1


class DeepQTrainer:
    def __init__(
        self,
        agent,
        env,
        replay_memory_size,
        episode_length,
        history_depth,
        batch_size,
        gamma,
        epsilon_scheduler,
        show_summary=True,
        action_name_map=None,
    ):
        self.action_tracker = ActionTracker(action_name_map)

        self.agent = agent
        self.env = env
        self.replay_memory_size = replay_memory_size
        self.eplength = episode_length
        self.hdepth = history_depth
        self.batch_size = batch_size
        if not hasattr(gamma_scheduler, "__iter__"):
            gamma_scheduler = Repeater(gamma_scheduler)
        if not hasattr(epsilon_scheduler, "__iter__"):
            epsilon_scheduler = Repeater(epsilon_scheduler)
        self.epsilon_get = iter(epsilon_scheduler)
        self.gamma_get = iter(gamma_scheduler)
        print("Training with")
        print("  Memory size    ", self.replay_memory_size)
        print("  Episode length ", self.eplength)
        print("  History depth  ", self.hdepth)
        print("  Batch size     ", self.batch_size)
        print("  Reward discount", self.gamma_get)
        print("  Epsilon        ", self.epsilon_get)
        print("Agent")
        print(self.agent)

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=0.0001)
        self.lossfn = torch.nn.MSELoss()

    def progress_summary(self, e, n, lens, loss, r, clear=True):
        avg_len = sum(lens) / len(lens)
        print(f"{e} / {n}"
            + f" : Avg Episode length {avg_len:.1f}/{self.eplength}"
            + f" : Loss {sum(loss) / len(loss):.3f}"
            + f" : Avg Reward {sum(r) / len(r):.2f}"
            + f" : epsilon {self.epsilon:.2f}"
            + f" : gamma {self.gamma:.2f}"
            + f"\nActions {self.action_tracker}"
        )
        if clear:
            lens.clear()
            self.action_tracker.reset()

    def train(self, n_episodes, show_every=32,):
        self.agent.eval()
        D = ReplayMemory(self.replay_memory_size)
        loss = []
        avg_len = []

        for episode in range(n_episodes):
            self.epsilon = next(self.epsilon_get)
            self.gamma = next(self.gamma_get)

            l, r, t = self.train_episode(D, epsilon)

            loss.append(l)
            avg_len.append(t)
            if episode % show_every == 0:
                self.progress_summary(episode, n_episodes, avg_len, l, r,)
                if episode > 128:
                    self.train_episode(D, render=True, eval=True)

        if avg_len:  # otherwise no final update needed
            self.progress_summary(episode, n_episodes, avg_len, l, r,)

        return loss

    def train_episode(self, D, render=False, eval=False,):
        x = self.env.reset()
        s = RollingState(self.hdepth, x)
        loss = []
        rewards = []

        epsilon = 0.05 if eval else self.epsilon

        for t in range(self.eplength):
            if render:
                self.env.render()
                sleep(.04)
            if random.random() < epsilon:
                # epsilon greedy action selection
                a = self.env.action_space.sample()
            else:
                agent_input = s.next[None]
                with torch.no_grad():
                    a = torch.argmax(self.agent(agent_input)).item()
                self.action_tracker.update(a)

            x, r, done, info = self.env.step(a)

            if not eval:
                s.update(x)

                D.store(s, a, r, done)
                l = self.train_batch(D).item()

                loss.append(l)
                rewards.append(r)

            if done: break  # after last batch train?

        return loss, rewards, t+1

    def train_batch(self, D):
        batch = D.sample(self.batch_size)  # N, 3 (s, a, r)
        s, *ard = zip(*batch)
        a, r, done = map(lambda t: tensor(t,device=device), ard)  # N  x3
        s, sp = zip(*map(RollingState.time_split, s))  # N, H, S  x2
        s, sp = torch.stack(s), torch.stack(sp)  # N, H, S  x2

        with torch.no_grad():
            out = self.agent(sp)  # N, A
        out, _ = torch.max(out, dim=1)  # N
        y = r + self.gamma * out * (done.logical_not())

        self.agent.train()
        self.optimizer.zero_grad()
        yhat = self.agent(s)
        yhat = yhat.gather(1, a[:,None]).squeeze()
        loss = self.lossfn(yhat, y)
        loss.backward()
        self.optimizer.step()
        self.agent.eval()

        if VERBOSE and 0:
            print()
            print('r', sum(r).item());print(r)
            print('done', sum(done).item());print(done)
            print('y');print(y)
            print('s[0]');print(s[0])
            print('sp[0]');print(sp[0])
            print('yhat');print(yhat)
            print('loss');print(loss)

        return loss


class RollingState(torch.Tensor):
    @staticmethod
    def __new__(cls, time_depth, s, *args, **kwargs):
        if isinstance(s, (list, tuple)):
            s = np.array(s)
        # s maybe time slice to initialize rolling state, or (s, *args)
        # may be shape to initialize emtpy (zero'd) rolling state
        if torch.is_tensor(s) or isinstance(s, np.ndarray):
            self = super().__new__(cls, time_depth+1, *s.shape, *args, **kwargs).to(device)
            self.fill_(.0)
            self.update(s)
        else:  # assume (s, *args) holds rest of shape  (shape[1:])
            self = super().__new__(cls, time_depth+1, s, *args, **kwargs).to(device)
            self.fill_(.0)

        return self

    def update(self, s: np.ndarray) -> torch.Tensor:
        self[0, ...].copy_(torch.from_numpy(s))
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

    def copy(self, *args, **kwargs):
        shape = self.shape
        new = RollingState(shape[0]-1, *shape[1:], *args, **kwargs)
        new.copy_(self)
        return new


class ReplayMemory(deque):
    Experience = namedtuple("Experience", ('s', 'a', 'r', 'done'))

    def __init__(self, l=None, maxlen=None):
        if l is None:
            l = []
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
        copy_state=True,
    ):
        if copy_state:
            s = s.copy()
        expr = ReplayMemory.Experience(s, a, r, done)
        self.append(expr)
        return self

    def sample(self, k, *args, **kwargs):
        return random.choices(self, k=k, *args, **kwargs)

    def careful_sample(self, k, *args, with_replacement=False, **kwargs):
        if len(self) < k or with_replacement:
            # random.choices automatically samples w/ replacement
            return random.choices(self, k=k, *args, **kwargs)
        return random.sample(self, k=k, *args, **kwargs)


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
        return str(C) + ' ' + ' '.join(f"{a}:{c/C:.2f}" for a,c in items)

