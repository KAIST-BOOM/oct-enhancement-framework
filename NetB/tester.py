import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from model import Generator, Discriminator
import utils
import data_loader
import time
import os
import pytorch_ssim
import pytorch_msssim
from scipy import io

model_name = '20230104'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


'''Dataset setting'''
path = '/home/leewj/projects/projects/oct_enhancement/degradation_test/[1]/image'

trainloader = data_loader.get_loader(dataset_path=path, phase='train', shuffle=True, batch_size=1, num_workers=1)
valloader = data_loader.get_loader(dataset_path=path, phase='val', shuffle=True, batch_size=1, num_workers=1)
testloader = data_loader.get_loader(dataset_path=path, phase='test', shuffle=True, batch_size=1, num_workers=1)

netG = Generator()
netG = nn.DataParallel(netG, device_ids=[0], output_device=0)
netD = Discriminator()
netD = nn.DataParallel(netD, device_ids=[0], output_device=0)
netG.to(device)
netD.to(device)

Gcheckpoint = torch.load('output/netG_%s.pth' % (model_name))
netG.load_state_dict(Gcheckpoint['State_dict'])

Dcheckpoint = torch.load('output/netD_%s.pth' % (model_name))
netD.load_state_dict(Dcheckpoint['State_dict'])

TRIAL = 50
saving = True
vis = False

for trial in range(1, TRIAL + 1):
    netG.eval()
    tic = time.time()
    with torch.no_grad():
        data, target = next(iter(testloader))
        batch_size = data.size(0)
        data, target = data.to(device), target.to(device)
        real_img = Variable(target)
        fake_img = netG(data)

        test_ssim = pytorch_ssim.ssim(fake_img, real_img).item()
        test_msssim = pytorch_msssim.msssim(fake_img, real_img, normalize=True).item()

        if vis:
            result_show = torchvision.utils.make_grid((torch.stack((data[0, :], real_img[0, :], fake_img[0, :]), dim=0)))
            utils.imshow(result_show.cpu().squeeze(), title='')

        if saving:
            index = time.strftime('%m%d%H%M%S', time.localtime(time.time()))
            out_path = 'output_image/' + model_name + '/%s'%(index)

            if not os.path.exists(out_path):
                os.makedirs(out_path)
            result = {
                'input' : data[0,0,:].squeeze().detach().cpu().numpy()*20+96,
                'real' : real_img.squeeze().detach().cpu().numpy()*20+96,
                'fake' : fake_img.squeeze().detach().cpu().numpy()*20+96,
            }
            io.savemat(out_path + '/result.mat', result)

        print('Trial:[%d], Time:[%.2f], SSIM:[%.4f], MSSSIM:[%.4f]'%(trial, time.time()-tic, test_ssim, test_msssim))