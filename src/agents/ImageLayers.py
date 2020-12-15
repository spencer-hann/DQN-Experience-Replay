import torch

from torch import tensor
from torch.nn import functional as F
from torchvision import transforms as T

from PIL import Image


# TODO: make this a class for easier saving (torch.save)
def ImageEater(crop=(0,-1), size=84, interpolation=Image.NEAREST):
    size = (size, size)
    c1, c2 = crop
    def crop(x):
        if len(x.shape) == 4:  # N, H, W, C
            return x[:, c1:c2]
        return x[c1:c2]  # H, W, C

    def show_shape(name):
        def _show_shape(x):
            print(name)
            print(x.shape, flush=True)
            return x
        return _show_shape

    return T.Compose([
        #T.Lambda(show_shape('input')),
        T.Lambda(crop),
        #T.Lambda(show_shape('crop')),
        T.ToTensor(), # (H,W,C) -> tensor((C,H,W))
        #T.Lambda(show_shape('ToTensor')),
        T.Grayscale(),
        #T.Lambda(show_shape('Grayscale')),
        T.Resize(size, interpolation=interpolation),
        #T.Lambda(show_shape('Resize')),
        # squeeze channel to be stacked with other timeslices
        T.Lambda(torch.squeeze),
        #T.Lambda(show_shape('squeeze')),
    ])


def prep_frame(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not torch.is_tensor(x):
        x = tensor(x)

    return x.unsqueeze(0)

