import torch
import random
import numpy as np

from torch import tensor
from collections import deque, namedtuple, defaultdict
from time import sleep

from ..Schedulers import Repeater
from ...Device import device
from .ReplayMemory import ReplayMemory
from .Utils import RollingState, ActionTracker


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
        weight_decay=0.0,
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
        self.epsilon = iter(epsilon_scheduler)
        print("Training with")
        print("  Memory size    ", self.replay_memory_size)
        print("  Episode length ", self.eplength)
        print("  History depth  ", self.hdepth)
        print("  Batch size     ", self.batch_size)
        print("  Reward discount", self.gamma_get)
        print("  Epsilon        ", self.epsilon)
        print("  Learning Rate  ", lr)
        print("  Weight Decay   ", weight_decay)
        print("Agent")
        print(self.agent)
        print("N Parameters")
        print(sum(map(len, map(torch.flatten, self.agent.parameters()))))

        self.lossfn = torch.nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(
            self.agent.parameters(), lr=lr, weight_decay=weight_decay
        )

    def progress_summary(self, e, n, lens, loss, r, clear=True, one_line=False):
        avg_len = sum(lens) / len(lens)
        print(f"{e} / {n}"
            + f" : Avg Episode length {round(avg_len)}/{self.eplength}"
            + f" : Loss {sum(loss) / len(loss):.3f}"
            + f" : Avg Total Reward {r:.2f}"
            + f" : epsilon {next(self.epsilon):.2f}"
            + f" : gamma {next(self.gamma_get):.3f}"
            + ('' if one_line else '\n')
            + f"Actions {self.action_tracker}"
        )
        if clear:
            lens.clear()
            self.action_tracker.reset()

    def train(self, n_episodes, stop_after=float('inf'), show_every=64, render_after=4000):
        self.agent.eval()  # explicitly set train in train_batch()
        D = ReplayMemory(self.replay_memory_size)
        loss, avg_len, rewards, Q = [], [], [], []

        for episode in range(n_episodes):
            l, r, t, q = self.train_episode(D,)

            loss.append(l)
            avg_len.append(t)
            rewards.append(r)
            Q.append(q)

            if episode % show_every == 0:
                avg_r = sum(rewards[-show_every:]) / show_every
                self.progress_summary(episode, n_episodes, avg_len, l, avg_r,)
                avg_len = []
                self.agent.save(f"current_agent_{self.agent.name}.pt")
                if episode > 40:
                    #self.agent.render.on()
                    eval_r, eval_q = self.eval(n_episodes=4, render=(episode > render_after))
                    self.agent.render.off()
                    print(f"Eval Reward {eval_r} : Eval Q {eval_q:.4f}")

            if r > stop_after:
                if show_every > 0:
                    print(f"Reward > {stop_after}")
                    print("Stopping...")
                break

        if avg_len and show_every > 0:  # otherwise no final update needed
            self.progress_summary(episode, n_episodes, avg_len, l, r,)

        return loss, rewards, Q

    def eval(self, n_episodes, epsilon=0.05, render=False, render_sleep=0.03,):
        Q = R = 0
        for e in range(n_episodes):
            r, q = self.eval_episode(epsilon, render, render_sleep)
            Q += sum(q)
            R += sum(r)
        return R / n_episodes, Q / n_episodes

    def eval_episode(self, epsilon=0.05, render=False, render_sleep=0.03,):
        s = self.start_episode()
        Q = []
        R = []

        last_q = last_r = None

        train_mode = self.agent.training
        self.agent.eval()
        for t in range(self.eplength):
            _, q, r, done = self.step(s.next, epsilon)

            #if last_r:
            #    print(round(q, 4))
            #if r:
            #    print(
            #        'a', _, self.action_tracker.action_map[_],
            #        '\tr', r,
            #        '\tq', round(last_q,4), round(q,4),
            #        end = ' ',
            #    )

            #last_q = q
            #last_r = r

            Q.append(q)
            R.append(r)

            if render:
                sleep(render_sleep)
                self.env.render()

            if done:
                break

        if train_mode:  # reset to original state
            self.agent.train()

        return R, Q

    def start_episode(self):
        x = self.env.reset()
        x = self.agent.process_observation(x)
        s = RollingState(self.hdepth, x)
        return s

    def step(self, state, epsilon):
        # epsilon greedy action selection
        a, q = self.action_selection(state, epsilon)

        # gameplay
        x, r, done, info = self.env.step(a)

        # prep for next step
        x = self.agent.process_observation(x)
        state.update(x)

        return a, q, r, done

    ## gym environments already use stocastic frame skipping
    #def k_step(self, a, s, k=4):  # 4 in arxiv:1312.5602v1
    #    r = 0
    #    for k in range(4):
    #        #print('k', k, 'a', a, self.action_tracker.action_map[a])
    #        x, _r, done, info = self.env.step(a)
    #        r += _r
    #        x = self.agent.process_observation(x)
    #        s.update(x)
    #        if done: return r, done
    #    return r, done

    def train_episode(self, D,):
        s = self.start_episode()
        loss = []
        total_reward = 0
        Q = 0

        for t in range(self.eplength):
            # epsilon greedy action selection
            a, q, r, done = self.step(s.next, next(self.epsilon))

            Q += q
            total_reward += r

            D.store(s, a, r, done)
            l = self.train_batch(D)
            loss.append(l.item())

            if done: break

        return loss, total_reward, t+1, Q

    def train_batch(self, D):
        if len(D) < self.batch_size:
            return tensor(0.0)
        batch = D.sample(self.batch_size)  # N, 3 (s, a, r)
        s, *ard = zip(*batch)
        a, r, done = map(lambda t: tensor(t,device=device), ard)  # N  x3
        s, sp = zip(*map(RollingState.time_split, s))  # N, H, S  x2
        s, sp = torch.stack(s), torch.stack(sp)  # N, H, S  x2

        with torch.no_grad():
            out = self.agent(sp)  # N, A
        out, _ = torch.max(out, dim=1)  # N
        y = r + next(self.gamma_get) * out * (done.logical_not())

        self.agent.train()
        self.optimizer.zero_grad()
        yhat = self.agent(s)
        yhat = yhat.gather(1, a[:,None])#[1]#.squeeze()
        yhat = yhat.squeeze()
        loss = self.lossfn(yhat.T, y)
        loss.backward()
        self.optimizer.step()

        self.agent.eval()
        if 0:# and random.random() < .01:
            print()
            print('r', sum(r).item());print(r)
            print('done', sum(done).item());print(done)
            print('y');print(y)
            print('s[0]');print(s[0])
            print('sp[0]');print(sp[0])
            print('yhat');print(yhat)
            print('loss');print(loss)

        return loss

    def action_selection(self, state, epsilon):
        a, q = self.agent.select_action(
            state[None], self.env.action_space, epsilon
        )
        a = a.item()
        self.action_tracker.update(a)
        return a, q.item()

