import torch
import random
import numpy as np

from torch import tensor
from collections import deque, namedtuple, defaultdict
from time import sleep

from .Schedulers import Repeater
from ..Device import device


epmax = 0


class DeepQTrainer:
    def __init__(
        self,
        agent,
        env,
        replay_memory_size,
        episode_length,
        history_depth,
        batch_size,
        gamma_scheduler,
        epsilon_scheduler,
        lr=0.001,
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
        self.gamma_get = iter(gamma_scheduler)
        self.epsilon_get = iter(epsilon_scheduler)
        self.lr = lr
        print("Training with")
        print("  Memory size    ", self.replay_memory_size)
        print("  Episode length ", self.eplength)
        print("  History depth  ", self.hdepth)
        print("  Batch size     ", self.batch_size)
        print("  Reward discount", self.gamma_get)
        print("  Epsilon        ", self.epsilon_get)
        print("  Learning Rate  ", self.lr)
        print("Agent")
        print(self.agent)

        self.optimizer = torch.optim.RMSprop(self.agent.parameters(), lr=self.lr)
        self.lossfn = torch.nn.MSELoss()

    def progress_summary(self, e, n, lens, loss, r, clear=True):
        avg_len = sum(lens) / len(lens)
        print(f"{e} / {n}"
            + f" : Avg Episode length {avg_len:.1f}/{self.eplength}"
            + f" : Loss {sum(loss) / len(loss):.3f}"
            + f" : Avg Total Reward {r:.2f}"
            + f" : epsilon {self.epsilon:.2f}"
            + f" : gamma {self.gamma:.2f}"
            + f"\nActions {self.action_tracker}"
        )
        if clear:
            lens.clear()
            self.action_tracker.reset()

    def train(self, n_episodes, show_every=32, render_after=4000):
        #self.agent.eval()  # explicitly set train in self.train_batch
        D = ReplayMemory(self.replay_memory_size)
        loss, avg_len, avg_r, r = [], [], [], []

        for episode in range(n_episodes):
            self.epsilon = next(self.epsilon_get)
            self.gamma = next(self.gamma_get)

            l, r, t = self.train_episode(D,)

            loss.append(l)
            avg_len.append(t)
            avg_r.append(sum(r))
            if episode % show_every == 0:
                avg_r = sum(avg_r) / len(avg_r)
                self.progress_summary(episode, n_episodes, avg_len, l, avg_r,)
                avg_r = []
                #self.agent.save(f"current_agent_{self.agent.name}.pt")
                if episode > render_after:
                    self.train_episode(D, render=True, eval=True)

        if avg_len:  # otherwise no final update needed
            self.progress_summary(episode, n_episodes, avg_len, l, r,)

        return loss, sum(r)

    def train_episode(self, D, render=False, eval=False,):
        global epmax
        x = self.env.reset()
        #x = self.agent.process_observation(x)
        x = torch.from_numpy(x)
        s = RollingState(self.hdepth, x)
        loss = []
        rewards = []

        epsilon = 0.05 if eval else self.epsilon

        for t in range(self.eplength):
            if random.random() < epsilon:
                # epsilon greedy action selection
                a = self.env.action_space.sample()
            else:
                agent_input = s.next[None]
                with torch.no_grad():
                    a = torch.argmax(self.agent(agent_input)).item()
                self.action_tracker.update(a)

            x, r, done, info = self.env.step(a)
            r -= done
            if not eval:
                #x = self.agent.process_observation(x)
                x = torch.from_numpy(x)
                s.update(x)

                D.store(s, a, r, done)
                #print('t', t)
                l = self.train_batch(D)

                loss.append(l.item())
                rewards.append(r)

            if render:
                sleep(.03)
                self.env.render()

            if done: break

        return loss, rewards, t+1

    def train_batch(self, D):
        batch = D.careful_sample(self.batch_size)  # N, 3 (s, a, r)
        s, *ard = zip(*batch)
        a, r, done = map(lambda t: tensor(t,device=device), ard)  # N  x3
        s, sp = zip(*map(RollingState.time_split, s))  # N, H, S  x2
        s, sp = torch.stack(s), torch.stack(sp)  # N, H, S  x2

        with torch.no_grad():
            out = self.agent(sp)  # N, A
        out, _ = torch.max(out, dim=1)  # N
        y = r + self.gamma * out * (done.logical_not())

        #get_params = lambda a: next(a.parameters()).clone()#.detach()

        #p1 = get_params(self.agent)
        #p1[...] = get_params(self.agent)
        #print('p1', id(p1))
        #print(p1)
        #self.agent.train()
        self.optimizer.zero_grad()
        yhat = self.agent(s)
        yhat = yhat.gather(1, a[:,None])#[1]#.squeeze()
        yhat = yhat.squeeze()
        loss = self.lossfn(yhat.T, y)
        loss.backward()
        #print("params")
        #print(next(self.agent.parameters()))
        #print("params, grad")
        #print(next(self.agent.parameters()).grad)
        self.optimizer.step()
        #p2 = get_params(self.agent)

        #self.agent.eval()
        if 0:# and random.random() < .01:
            print()
            print('r', sum(r).item());print(r)
            print('done', sum(done).item());print(done)
            print('y');print(y)
            print('s[0]');print(s[0])
            print('sp[0]');print(sp[0])
            print('yhat');print(yhat)
            print('loss');print(loss)

        #print('p2', id(p2))
        #print(p2)
        #if not hasattr(self, '_count'):
        #    self._count = 1
        #else:
        #    self._count += 1
        #    if self._count > 10:
        #        assert p1 is not p2
        #        assert p1[0] is not p2[0]
        #        eq = (p1 != p2).flatten()
        #        assert all(eq), sum(eq) / len(eq)

        return loss

    ## gym environments use stocastic frame skipping already
    #def k_step(self, a, s, k=4):  # 4 in arxiv:1312.5602v1
    #    r = 0
    #    for k in range(4):
    #        x, _r, done, info = self.env.step(a)
    #        r += _r
    #        x = self.agent.process_observation(x)
    #        s.update(x)
    #        if done: return r, done
    #    return r, done


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

