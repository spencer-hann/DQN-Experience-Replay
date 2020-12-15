import gym
import time


from src.UserIO.arrowkeys import get_arrow_key


env = gym.make('MountainCar-v0')
env = gym.make('CartPole-v0')
env = gym.make('Breakout-ram-v0')
env = gym.make('Breakout-v0')
# save video of Agent interacting in this environment
#env = gym.wrappers.Monitor(env, './video/', force = True)


env.close()
for episode in range(4):
    observation = env.reset()
    t = done = 0
    while not done:
       t += 1
       env.render()
       screen = env.render("rgb_array")
       assert (observation == screen).all()
       time.sleep(.06)
       action = env.action_space.sample()
       action = get_arrow_key({'right':2, 'left':3, 'down':1})
       #print(observation, action)
       observation, reward, done, info = env.step(action)
       print(reward)

    print("Episode finished after {} timesteps".format(t+1))

env.close()



def train_batch(model, lossfn, optim, batch):
    wav, label = batch
    optim.zero_grad()
    out = model(wav.to(device))
    loss = lossfn(out, label.to(device))
    loss.backward()
    optim.step()
    return loss

