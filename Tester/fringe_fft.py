import torch
import numpy as np
import math
from scipy.fft import fft
import scipy.signal

class fringe_FFT():
    def __init__(self, nfft):
        self.nfft = nfft
        self.window = np.hanning(1300)

    def transform(self, data, phase):
        Fxx = fft(data, n=self.nfft, axis=1)
        mag = 10*np.log10(np.abs(Fxx)**2)
        mag = torch.tensor(mag).float()

        # Frequency shift -n_fft/4
        mag = torch.cat([mag[:,1024:2048], mag[:,0:1024]], dim=1)

        # Background region remove
        # mag = mag[:, 1023-649:1024+650]

        # mag = torch.reshape(mag, [mag.size()[0], 1, mag.size()[1], 1])
        return mag
