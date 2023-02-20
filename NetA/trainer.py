import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import torchvision
import pandas as pd

from loss import GeneratorLoss
from model import Generator, Discriminator
import utils
import data_loader
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tensorboardX import SummaryWriter
import warnings
import random
import os

def warn_func():
    warn_message = 'warn_func() is deprecated, use new_function() instead'
    warnings.warn(warn_message)

warnings.filterwarnings(action='ignore')
warn_func()

model_name = '20230104'

utils.get_clear_tensorboard()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(logdir="log_tensorboard/")

'''Transform settings'''

'''Dataset setting'''
path = '/projects/oct_enhancement/k_space/dataset'

trainloader = data_loader.get_loader(dataset_path=path, phase='train', shuffle=True, batch_size=24, num_workers=10)
valloader = data_loader.get_loader(dataset_path=path, phase='val', shuffle=True, batch_size=24, num_workers=10)

netG = Generator()
netD = Discriminator()

generator_criterion = GeneratorLoss()
discriminator_criterion = nn.BCELoss()

netG = nn.DataParallel(netG, device_ids=[0,1,2,3], output_device=0)
netD = nn.DataParallel(netD, device_ids=[0,1,2,3], output_device=0)

netG.to(device)
netD.to(device)
generator_criterion.to(device)
discriminator_criterion.to(device)

optimizerG = optim.Adam(netG.parameters(), lr=1e-3, weight_decay=5e-4)
optimizerD = optim.Adam(netD.parameters(), lr=1e-3)

schedulerG = ReduceLROnPlateau(optimizerG, mode='min', factor=0.1, patience=100, min_lr = 1e-7, verbose=False)

results = {'d_loss': [], 'g_loss': [],'d_score': [], 'g_score': [], 'g_loss_mse' : [], 'g_loss_l1' : [],
           'g_loss_grad' : [], 'g_loss_ad' : [], 'd_loss_real' : [], 'd_loss_fake' : [],
           'val_loss' : [], 'val_mse' : [], 'val_l1' : [], 'val_grad' : []}

NUM_EPOCHS = 700

for epoch in range(1, NUM_EPOCHS + 1):
    train_bar = tqdm(trainloader)
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    generator_loss = {'MSE' : 0, 'L1' : 0, 'GRAD' : 0, 'Adversarial' : 0}
    discriminator_loss = {'REAL' : 0, 'FAKE' : 0}

    netG.train()
    netD.train()
    tic = time.time()
    with torch.autograd.set_detect_anomaly(True):
        for data_stft, data, target in train_bar:
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            real_labels = torch.ones((target.size(0), 1)).to(device)
            fake_labels = torch.zeros((target.size(0), 1)).to(device)

            g_update_first = True
            data_stft, data, target = data_stft.to(device), data.to(device), target.to(device)

            # Discriminator update
            optimizerD.zero_grad()
            out = netG(data_stft, data)

            real_out = netD(target)
            fake_out = netD(out)

            d_loss_real = discriminator_criterion(real_out, real_labels)
            d_loss_fake = discriminator_criterion(fake_out, fake_labels)

            d_loss = d_loss_real + d_loss_fake

            if epoch < 100:
                if epoch%10 == 0 and epoch == 1:
                    d_loss.backward(retain_graph=True)
                    optimizerD.step()
            else:
                d_loss.backward(retain_graph=True)
                optimizerD.step()

            # Generator update
            optimizerG.zero_grad()

            out = netG(data_stft, data)
            fake_out = netD(out)

            adversarial_loss = discriminator_criterion(fake_out, real_labels)
            mse_loss, l1_loss, grad_loss = generator_criterion(out, target)

            g_loss = mse_loss * 0.6 + l1_loss + grad_loss * 0.9
            g_loss += adversarial_loss * 1e-4
            g_loss.backward(retain_graph=True)
            optimizerG.step()

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.mean().item() * batch_size
            running_results['g_score'] += fake_out.mean().item() * batch_size

            generator_loss['MSE'] += (mse_loss*0.3).item() * batch_size
            generator_loss['L1'] += (l1_loss*0.7).item() * batch_size
            generator_loss['GRAD'] += (grad_loss * 0.9).mean().item() * batch_size
            generator_loss['Adversarial'] += (adversarial_loss * 1e-4).mean().item() * batch_size

            discriminator_loss['REAL'] += d_loss_real.mean().item() * batch_size
            discriminator_loss['FAKE'] += d_loss_fake.mean().item() * batch_size

            train_bar.set_description(
                desc='[%d/%d] L_D:%.3f L_G:%.3f D(x):%.2f D(G(z)):%.2f, Time:%.2f s, LR:%.5f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes'],
                time.time() - tic,
                optimizerG.param_groups[0]['lr'])
            )

        if epoch % 100 == 0 or epoch == 1:
            utils.show(data_stft[0,:], data[0,:], target[0,:], out[0,:], title=str(epoch))

    schedulerG.step(g_loss)
    netG.eval()
    with torch.no_grad():
        val_bar = tqdm(valloader)
        valing_results = {'MSE': 0, 'L1': 0, 'GRAD': 0, 'loss': 0, 'batch_sizes': 0}
        for data_stft, data, target in val_bar:
            batch_size = target.size(0)
            valing_results['batch_sizes'] += batch_size

            data_stft, data, target = data_stft.to(device), data.to(device), target.to(device)

            out = netG(data_stft, data)

            mse_loss, l1_loss, grad_loss = generator_criterion(out, target)
            g_val_loss = mse_loss * 0.6 + l1_loss + grad_loss * 0.9

            valing_results['MSE'] += (mse_loss*0.3).item() * batch_size
            valing_results['L1'] += (l1_loss*0.7).item() * batch_size
            valing_results['GRAD'] += (grad_loss*0.9).mean().item() * batch_size

            valing_results['loss'] += g_val_loss.data.mean() * batch_size

            val_bar.set_description(
                desc='[Validating results] Loss: %.4f' % (valing_results['loss'] / valing_results['batch_sizes']))


    # save loss\scores\psnr\ssim
    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
    results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
    results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])

    results['g_loss_mse'].append(generator_loss['MSE'] / running_results['batch_sizes'])
    results['g_loss_l1'].append(generator_loss['L1'] / running_results['batch_sizes'])
    results['g_loss_grad'].append(generator_loss['GRAD'] / running_results['batch_sizes'])
    results['g_loss_ad'].append(generator_loss['Adversarial'] / running_results['batch_sizes'])

    results['d_loss_real'].append(discriminator_loss['REAL'] / running_results['batch_sizes'])
    results['d_loss_fake'].append(discriminator_loss['FAKE'] / running_results['batch_sizes'])

    results['val_loss'].append(valing_results['loss'] / valing_results['batch_sizes'])
    results['val_mse'].append(valing_results['MSE'] / valing_results['batch_sizes'])
    results['val_l1'].append(valing_results['L1'] / valing_results['batch_sizes'])
    results['val_grad'].append(valing_results['GRAD'] / valing_results['batch_sizes'])

    writer.add_scalar('Loss/Loss D', running_results['d_loss'] / running_results['batch_sizes'], epoch)
    writer.add_scalar('Loss/Loss G', running_results['g_loss'] / running_results['batch_sizes'], epoch)

    writer.add_scalar('GENERATOR/MSE', generator_loss['MSE'] / running_results['batch_sizes'], epoch)
    writer.add_scalar('GENERATOR/L1', generator_loss['L1'] / running_results['batch_sizes'], epoch)
    writer.add_scalar('GENERATOR/GRAD', generator_loss['GRAD'] / running_results['batch_sizes'], epoch)
    writer.add_scalar('GENERATOR/Adversarial', generator_loss['Adversarial'] / running_results['batch_sizes'], epoch)

    writer.add_scalar('DISCRIMINATOR/RAEL', discriminator_loss['REAL'] / running_results['batch_sizes'], epoch)
    writer.add_scalar('DISCRIMINATOR/FAKE', discriminator_loss['FAKE'] / running_results['batch_sizes'], epoch)

    writer.add_scalar('Score/D(x)', running_results['d_score'] / running_results['batch_sizes'], epoch)
    writer.add_scalar('Score/D(G(z))', running_results['g_score'] / running_results['batch_sizes'], epoch)

    writer.add_scalar('ValingResults/MSE', valing_results['MSE'], epoch)
    writer.add_scalar('ValingResults/L1', valing_results['L1'], epoch)
    writer.add_scalar('ValingResults/GRAD', valing_results['GRAD'], epoch)

    writer.add_scalar('learning rate', optimizerG.param_groups[0]['lr'], epoch)
    # save model parameters
    if epoch > 1:
        utils.save_checkpoint(epoch, netG, optimizerG, 'output/netG_%s.pth' % (model_name))
        utils.save_checkpoint(epoch, netD, optimizerD, 'output/netD_%s.pth' % (model_name))

        train_results = pd.DataFrame(
            data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                  'Score_G': results['g_score'], 'g_loss_mse' : results['g_loss_mse'], 'g_loss_l1' : results['g_loss_l1'],
                  'g_loss_grad' : results['g_loss_grad'], 'g_loss_ad' : results['g_loss_ad'],
                  'd_loss_real' : results['d_loss_real'], 'd_loss_fake' : results['d_loss_fake'],
                  'val_loss' : results['val_loss'], 'val_mse' : results['val_mse'],
                  'val_l1' : results['val_l1'], 'val_grad' : results['val_grad']},
            index=range(1, epoch + 1))

        if not os.path.exists('statistics/%s' % (model_name)):
            os.makedirs('statistics/%s' % (model_name))

        train_results.to_csv('statistics/%s/%d_train_results.csv' % (model_name, epoch), index_label='Epoch')

writer.close()