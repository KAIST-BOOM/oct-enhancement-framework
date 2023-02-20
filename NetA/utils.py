import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.signal

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

def show(data_stft, data, target, out, title = None):
    data_stft = data_stft.squeeze().detach().cpu().numpy()*35 + 35
    data = data.squeeze().detach().cpu().numpy() * 25 + 100
    target = target.squeeze().detach().cpu().numpy()*25 + 100
    out = out.squeeze().detach().cpu().numpy()*25 + 100

    plt.subplot(1, 4, 1)
    for i in range(0, np.shape(data_stft)[0]):
        for j in range(0, np.shape(data_stft)[2]):
            plt.plot(data_stft[i,:,j], np.arange(1, 1024+1, step=1))
    plt.gca().invert_yaxis()
    plt.title([title + ' input_stft'])
    plt.grid(b=True, axis='both')

    plt.subplot(1, 4, 2)
    for i in range(0, np.shape(data)[0]):
        plt.plot(data[i,:], np.arange(1, 1024 + 1, step=1))
    plt.gca().invert_yaxis()
    plt.title([title + ' input'])
    plt.xlim(70, 130)
    plt.grid(b=True, axis='both')

    plt.subplot(1, 4, 3)
    for i in range(0, np.shape(target)[0]):
        plt.plot(target[i,:], np.arange(1, 1024 + 1, step=1))
    plt.gca().invert_yaxis()
    plt.title([title + ' target'])
    plt.xlim(70, 130)
    plt.grid(b=True, axis='both')

    plt.subplot(1, 4, 4)
    for i in range(0, np.shape(out)[0]):
        plt.plot(out[i,:], np.arange(1, 1024 + 1, step=1))
    plt.gca().invert_yaxis()
    plt.title([title + ' out'])
    plt.xlim(70, 130)
    plt.grid(b=True, axis='both')

    plt.show()
    plt.pause(0.001)

def imshow(inp, title = None):
    # Set caxis
    cmin = 80/130
    cmax = 130/130
    inp = (inp.detach().numpy().transpose((1, 2, 0))*25 + 105)/130
    inp = np.clip(inp, cmin, cmax)
    inp = (inp-cmin)/(cmax-cmin)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
