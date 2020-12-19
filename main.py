import torch
import gym
import time
import random

from src.training.DeepQ import DeepQTrainer, RollingState
from src.training.Schedulers import LinearScheduler
from src.agents import VisualAgent, RamAgent
from src.Visualization import plot
from src.Device import device


#game_name = "CartPole-v0"
#game_name = "Breakout-ram-v0"
game_name = "Breakout-v0"
#game_name = "SpaceInvaders-v0"
#game_name = "MsPacman-v0"
env = gym.make(game_name)
print(f"GAME  {game_name}")


action_map = None
if hasattr(env.unwrapped, "get_action_meanings"):
    action_map = []
    for i, a in enumerate(env.unwrapped.get_action_meanings()):
        print(i, a)
        action_map.append(a)


history_depth = 4
input_size = history_depth*env.observation_space.shape[0]
#agent = RamAgent(input_size, env.action_space.n)
agent = VisualAgent(history_depth, env.action_space.n, size=84, crop=(30,200),)
agent.to(device)


epsilon = (1, .1, 2e6); print('epsilon', epsilon)
epsilon = LinearScheduler(*epsilon, start_delay=256)
gamma = (0, .9, 1e6); print('gamma', gamma)
gamma = LinearScheduler(*gamma, start_delay=4092)
#gamma = .9
#print("gamma no scheduler", gamma)
trainer = DeepQTrainer(
    agent,
    env.env,
    replay_memory_size=1_000_000,  # according to paper
    episode_length=2046*4,
    history_depth=history_depth,
    batch_size=32,
    gamma_scheduler=gamma,
    epsilon_scheduler=epsilon,
    action_name_map=action_map,
    lr=1e-5,
    weight_decay=1e-3,
)


reward="UNFINISHED"
try:
    loss, reward, Q = trainer.train(
        60000, show_every=64, render_after=float('inf')
    )
except KeyboardInterrupt:
    print()  # new line
finally:
    print("********************")
    print("* Training stopped *")
    print("********************")
    r = reward
    if r != "UNFINISHED":
        r = round(sum(reward[-64:]) / 64)
    save_name = f"{agent.name}_{game_name}_{r}.pt"
    print(f"Saving to {save_name}")
    agent.save(save_name)

if reward != "UNFINISHED":
    print("Creating plots...")
    plot.over_episodes(reward, "Total Episodic Reward", "Total Reward")
    plot.over_episodes(Q, "Total Expected Reward", "Value function (Q)")
    plot.loss(loss); del reward, Q, loss


env = env.env
# save video of Agent interacting in this environment
env = gym.wrappers.Monitor(
    env, f"./video/{game_name}", video_callable=lambda i: True, force = True
)
agent.eval()
for episode in range(10):
    input("Are you there? <Return>")

    observation = env.reset()
    observation = agent.process_observation(observation)
    state = RollingState(history_depth, observation)
    done = 0
    last_action = float('inf')
    t = 0
    while not done:
        t += 1
        env.render()
        time.sleep(.02)
        if random.random() < 0.05:
            print('*', end='')  # indicate epsilon random action
            action = env.action_space.sample()
        else:
            action = agent(state.next[None])
            action = torch.argmax(action).item()
        #print(observation, action, action_map[action])
        if action != last_action:
            last_action = action
            name = action_map[action] if action_map else action
            print(f"{name}({action})", end=' ')
        else:
            print('.', end='')
        observation, reward, done, info = env.step(action)
        observation = agent.process_observation(observation)
        state.update(observation)

    print(f"\nEpisode Length: {t}\n")

