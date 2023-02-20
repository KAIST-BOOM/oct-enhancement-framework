import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

def save_checkpoint(epoch, model, optimizer, filename):
    state = {
        'Epoch' : epoch,
        'State_dict' : model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }
    torch.save(state, filename)

def get_clear_tensorboard():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log_tensorboard')
    file_list = os.listdir(path)
    if file_list != None:
        for i in range(len(file_list)):
            os.remove(os.path.join(path, file_list[i]))


def imshow(inp, title = None):
    # Set caxis
    cmin = 80/130
    cmax = 1
    inp = (inp.detach().numpy().transpose((1, 2, 0))*25 + 105)/130
    inp = np.clip(inp, cmin, cmax)
    inp = (inp-cmin)/(cmax-cmin)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def imsave(data, index, title = None, model_name = None):
    out_path = 'output_image/' + model_name
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    cmin = 0
    cmax = 1
    data = data.detach().cpu().numpy().transpose((1, 2, 0)) + 80/130
    data = np.clip(data, cmin, cmax)
    data = (data-cmin)/(cmax-cmin)
    data = Image.fromarray(data)
    data.save(out_path + '/%s_%s.png' % (index, title))