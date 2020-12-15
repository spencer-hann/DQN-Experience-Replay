import torch
import gym
import time

from torch import nn

from src.training.DeepQ import DeepQTrainer, RollingState
from src.training.Schedulers import LinearScheduler
from src.agents import VisualAgent, RamAgent
from src.Device import device


game_name = "CartPole-v0"
#game_name = "Breakout-ram-v0"
#game_name = "Breakout-v0"
env = gym.make(game_name)
# save video of Agent interacting in this environment
#env = gym.wrappers.Monitor(env, './video/', force = True)
print(f"GAME  {game_name}")


action_map = None
if hasattr(env.unwrapped, "get_action_meanings"):
    action_map = []
    for i, a in enumerate(env.unwrapped.get_action_meanings()):
        print(i, a)
        action_map.append(a)


history_depth = 4
input_size = history_depth*env.observation_space.shape[0]
agent = RamAgent(input_size, env.action_space.n)
#agent = VisualAgent(history_depth, env.action_space.n, size=84, crop=(30,200),)
agent.to(device)


epsilon = LinearScheduler(1, .1, -0.0001, start_delay=256)
#gamma = LinearScheduler(0, .999, 0.004, start_delay=128)
gamma = .99
trainer = DeepQTrainer(
    agent,
    env.env,
    replay_memory_size=2046,
    episode_length=2046,
    history_depth=history_depth,
    batch_size=32,
    gamma_scheduler=gamma,
    epsilon_scheduler=epsilon,
    action_name_map=action_map,
    lr=1e-2,
)


reward="UNFINISHED"
try:
    loss, reward = trainer.train(4096*8, show_every=32, render_after=4000)
except KeyboardInterrupt:
    print()  # new line
finally:
    print("********************")
    print("* Training stopped *")
    print("********************")
    save_name = f"{agent.name}_{game_name}_{reward}.pt"
    print(f"Saving to {save_name}")
    agent.save(save_name)


#env = env.env
agent.eval()
for episode in range(10):
    input("Are you there? <Return>")

    observation = env.reset()
    observation = agent.process_observation(observation)
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
        observation = agent.process_observation(observation)
        state.update(observation)

