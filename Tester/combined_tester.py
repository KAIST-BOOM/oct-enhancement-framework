import torch.utils.data
import torch.nn as nn
import torchvision
import NetA
import NetB
import utils
import data_loader
import warnings
import os
import time
import scipy.io
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch
from scipy import io
from fringe_fft import fringe_FFT
from fringe_stft import fringe_STFT
import random

def warn_func():
    warn_message = 'warn_func() is deprecated, use new_function() instead'
    warnings.warn(warn_message)

warnings.filterwarnings(action='ignore')
warn_func()

'''
Combined tester:
oct signal enhancement for twice
1. stft2fft model
2. f space model
'''

fmodel_name = '20230104'
imodel_name = '20230104'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DatasFolder(Dataset):
    def __init__(self, root, phase):
        super(DatasFolder, self).__init__()
        assert phase in ["train", "val", "test"]
        self.phase = phase
        self.phase_folder = phase
        self.root = root

        # raw data size : 1300x1024 -> noverlap = 550
        # raw data size : 850x1024 -> noverlap = 580

        self.STFT_1300 = fringe_STFT(window='hann', nperseg=650, noverlap=550, nfft=2048, boundary='zeros', padded=True, axis=1)
        self.STFT_850 = fringe_STFT(window='hann', nperseg=425, noverlap=355, nfft=2048, boundary='zeros', padded=True, axis=1)

        self.FFT = fringe_FFT(nfft=2048)
        self.file_paths = os.listdir(os.path.join(self.root, "ground_truth", self.phase_folder))
        self.file_paths.sort()
        self.target_paths = os.listdir(os.path.join(self.root, "ground_truth", self.phase_folder))
        self.target_paths.sort()

        self.data_paths = []
        for i in range(len(self.file_paths)):
            data_path = (os.path.join(self.root, "data", self.phase_folder, self.file_paths[i]),
                         os.path.join(self.root, "ground_truth", self.phase_folder, self.target_paths[i]))
            self.data_paths.append(data_path)

    def get_image(self, index):
        center_v = 105.0
        max_v = 25.0

        center_sv = 35.0
        max_sv = 35.0

        file_path, target_path = self.data_paths[index]  # Random index
        name = self.file_paths[index]
        data_fringe = io.loadmat(file_path)
        target_fringe = io.loadmat(target_path)

        data_fringe = np.array(data_fringe['data'])
        target_fringe = np.array(target_fringe['ground_truth'])

        data_fringe = data_fringe.swapaxes(0,1)
        target_fringe = target_fringe.swapaxes(0, 1)

        if np.shape(data_fringe)[1] == 1300:
            data_stft = self.STFT_1300.transform(data_fringe)
        elif np.shape(data_fringe)[1] == 850:
            data_stft = self.STFT_850.transform(data_fringe)

        data = self.FFT.transform(data_fringe, phase='data')
        target = self.FFT.transform(target_fringe, phase='target')

        data_stft = torch.reshape(data_stft, [512,2,2048,14])
        data = torch.reshape(data, [512, 2, 2048, 1])
        target = torch.reshape(target, [512, 2, 2048, 1])

        data_stft = (data_stft - center_sv) / max_sv
        data = (data - center_v) / max_v
        target = (target - center_v) / max_v

        return data_stft, data, target, name

    def __len__(self):
        """Returns the total number of images."""
        return len(self.data_paths)

'''Dataset setting'''
path = '/projects/dataset/'
testloader = DatasFolder(root=path, phase='test')


netfG = NetA.Generator()
netfG = nn.DataParallel(netfG, device_ids=[0,1], output_device=0)

netiG = NetB.Generator()
netiG = nn.DataParallel(netiG, device_ids=[0,1], output_device=0)

netfG.to(device)
netiG.to(device)
fGcheckpoint = torch.load('/projects/NetA/output/netG_%s.pth' % (fmodel_name))
netfG.load_state_dict(fGcheckpoint['State_dict'])

iGcheckpoint = torch.load('/projects/NetB/output/netG_%s.pth' % (imodel_name))
netiG.load_state_dict(iGcheckpoint['State_dict'])

saving = True
vis = False
n_frames = 100

for frame in range(0, n_frames):
    netfG.eval()
    netiG.eval()
    with torch.no_grad():
        torch.cuda.empty_cache()
        data_stft, data, target, name = testloader.get_image(frame)

        data_stft = data_stft.squeeze(dim=0)
        data = data.squeeze(dim=0)
        target = target.squeeze(dim=0)

        data_stft = data_stft.to(device)
        data = data.to(device)

        tic = time.time()
        out = netfG(data_stft, data)
        toc1 = time.time()

        out = torch.reshape(out, [1,1,1024,2048])
        input_img = torch.reshape(data, [1024, 2048]).permute(1, 0)
        target_img = torch.reshape(target, [1024, 2048]).permute(1, 0)
        fake_img = out.permute(0,1,3,2)
        fake_img2 = netiG(fake_img)

        toc2 = time.time()

        print('[%d/%d] Elapsed time [A model] : %.2f, Elapsed time [A model + B model] : %.2f' %
              (frame, n_frames, toc1 - tic, toc2 - tic))

        if vis:
            result_show = torchvision.utils.make_grid(input_img)
            utils.imshow(result_show.cpu().squeeze(), title='Input')

            result_show = torchvision.utils.make_grid(fake_img)
            utils.imshow(result_show.cpu().squeeze(), title='Fake1')

            result_show = torchvision.utils.make_grid(fake_img2)
            utils.imshow(result_show.cpu().squeeze(), title='Fake2')

            result_show = torchvision.utils.make_grid(target_img)
            utils.imshow(result_show.cpu().squeeze(), title='target')


        if saving:
            index = name
            out_path = 'output_image/' + '220611' + '/%s'%(index)

            if not os.path.exists(out_path):
                os.makedirs(out_path)
            result = {
                'input' : input_img.squeeze().detach().cpu().numpy()*25+105,
                'fake1' : fake_img.squeeze().detach().cpu().numpy()*25+105,
                'fake2' : fake_img2.squeeze().detach().cpu().numpy()*25+105,
                'target': target_img.squeeze().detach().cpu().numpy()*25+105,
            }
            scipy.io.savemat(out_path + '/result.mat', result)
