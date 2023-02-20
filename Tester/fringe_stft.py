import torch
import numpy as np
from scipy.signal import stft, istft

class fringe_STFT():
    def __init__(self, window, nperseg, noverlap, nfft, boundary, padded, axis):
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.boundary = boundary
        self.padded = padded
        self.axis = axis

    def transform(self, fringe_data):
        f, t, Zxx = stft(fringe_data, window=self.window, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft,
                         boundary=self.boundary, padded=self.padded, axis=self.axis)

        # data normalization
        db_mag = 10*np.log10(np.abs(Zxx)**2)
        # phase = np.unwrap(np.angle(Zxx), axis=2)/500

        db_mag = torch.tensor(db_mag).float()
        # Frequency shift -n_fft/4

        db_mag = torch.cat([db_mag[:,1024:2048,:], db_mag[:,0:1024,:]], dim=1)
        # db_mag = db_mag[:,1023-649:1024+650,:]

        # phase = torch.tensor(phase).float().to(self.device)
        return db_mag

    def inverse(self, db_mag, phase):
        # data un_normalization
        db_mag = db_mag.detach()
        phase = phase.detach()*500

        mag = 10 ** db_mag

        re_Zxx = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase)).squeeze().cpu()
        t, re_fringe = istft(re_Zxx, window=self.window, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft, input_onesided=False,
                          boundary=self.boundary, freq_axis=self.axis)

        return re_fringe