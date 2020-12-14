import torch

from torch import tensor
from torch.nn import functional as F
from torchvision import transforms as T

from PIL import Image


def ImageEater(*args, **kwargs):
    return torch.nn.Sequential(
        T.Compose(
            gray = T.Grayscale(),
            resize = T.Resize(*args, **kwargs),  # 110 x 84 per paper
        )
    )


def prep_frame(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not torch.is_tensor(x):
        x = tensor(x)

    return x.unsqueeze(0)

