import numpy as np
import matplotlib.pyplot as plt


plt.style.use('ggplot')


def over_episodes(values, title, ylabel, style='C1'):
    plt.title(title)
    plt.xlabel("Episodes (Training Epochs)")
    plt.ylabel(ylabel)

    plt.plot(values, style)
    plt.show();

def loss(loss, smooth=7, *args, **kwargs):
    mean = lambda l: sum(l) / len(l)
    flat_loss = list(map(mean, loss))
    loss = np.empty(len(flat_loss))
    loss[:smooth] = flat_loss[0]
    for i in range(smooth, len(loss)):
        loss[i] = sum(flat_loss[i-smooth:i]) / smooth
    over_episodes(loss, "MSE Loss over time", "Loss", *args, **kwargs)

