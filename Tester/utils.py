import matplotlib.pyplot as plt
import numpy as np


def imshow(inp, title = None):
    # Set caxis
    cmin = 80/130
    cmax = 130/130
    #
    if title == 'Fake2':
        cmin = 85/130
        cmax = 135 / 130

    inp = (inp.detach().numpy().transpose((1, 2, 0))*25 + 105)/130
    inp = np.clip(inp, cmin, cmax)
    inp = (inp-cmin)/(cmax-cmin)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
