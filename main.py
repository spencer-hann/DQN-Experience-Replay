import torch
import gym
import time

from torch import nn

from training.DeepQ import DeepQTrainer, RollingState
from training.Schedulers import LinearScheduler
from Device import device


#env = gym.make('CartPole-v0')
#env = gym.make('Breakout-ram-v0')
env = gym.make('Breakout-v0')
# save video of Agent interacting in this environment
#env = gym.wrappers.Monitor(env, './video/', force = True)


action_map = []
for i, a in enumerate(env.unwrapped.get_action_meanings()):
    print(i, a)
    action_map.append(a)


history_depth = 2
input_size = history_depth*env.observation_space.shape[0]
print(env.observation_space)
print(env.observation_space.shape)
print(env.observation_space.shape[0])
print(input_size)
#agent = RamSequetialAgent(input_size, env.action_space.n)
agent = VisualAgent(input_size, env.action_space.n)
agent.to(device)


epsilon = LinearScheduler(.05, 0.0002, start_delay=256)
gamma = LinearScheduler(.0, .999, 0.001, start_delay=256)
trainer = DeepQTrainer(
    agent,
    env.env,
    replay_memory_size=2046,
    episode_length=2046,
    history_depth=history_depth,
    batch_size=64,
    gamma_scheduler=gamma,
    epsilon_scheduler=epsilon,
    action_name_map=action_map,
)

try:
    trainer.train(4096*8, 64)
except KeyboardInterrupt:
    print("Training stopped")


#env = env.env
for episode in range(10):
    input("Are you there? <Return>")

    observation = env.reset()
    state = RollingState(history_depth, observation)
    done = 0
    last_action = float('inf')
    while not done:
        env.render()
        time.sleep(.04)
        action = agent(state.now[None])
        action = torch.argmax(action).item()
        #print(observation, action, action_map[action])
        if action != last_action:
            print(f", {action_map[action]}({action})", end='')
        else:
            print('.', end='')
        observation, reward, done, info = env.step(action)
        state.update(observation)
    time.sleep(1)

