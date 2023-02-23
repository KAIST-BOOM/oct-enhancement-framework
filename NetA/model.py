import math
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, n_basic_block=4):
        ''' 4-Aline's Amplification of Spectrogram Generator '''
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU()
        )
        basic_block_layer = []

        for _ in range(n_basic_block):
            basic_block_layer += [ResidualInResidualDenseBlock()]
        self.baskc_block = nn.Sequential(*basic_block_layer)

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3,4), stride=1, padding=(1, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 4), stride=1, padding=(1, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 4), stride=1, padding=(1, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        # 64x1300x5

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        # 64x1300x1

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.skip_connection = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU()
        )

    def forward(self, x, x2):
        x2 = self.skip_connection(x2)
        x1 = self.layer1(x)
        x = self.baskc_block(x1)
        x = self.layer2(x)
        x = self.layer3(x+x1)
        x = self.layer4(x)
        x = self.layer5(x + x2)
        return x

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),  # input is 1300
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 650
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 325
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 125
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 64
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 32
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        # 20992 : 1300 full
        # 10752 : 650 crop
        self.classifier = nn.Sequential(
            nn.Linear(16384, 100),
            nn.PReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return torch.sigmoid(out)

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels=64, growth_channels=32, scale_factor=0.2):
        super(ResidualDenseBlock, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels, growth_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(growth_channels),
            nn.PReLU()
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(in_channels+1*growth_channels, growth_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(growth_channels),
            nn.PReLU()
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(in_channels+2*growth_channels, growth_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(growth_channels),
            nn.PReLU()
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(in_channels+3*growth_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.PReLU()
        )

        self.scale_factor = scale_factor

    def forward(self, x):
        l1 = self.l1(x)
        l2 = self.l2(torch.cat((x, l1), dim=1))
        l3 = self.l3(torch.cat((x, l1, l2), dim=1))
        l4 = self.l4(torch.cat((x, l1, l2, l3), dim=1))
        return l4.mul(self.scale_factor) + x

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, in_channels=64, growth_channels=32, scale_factor=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.l1 = ResidualDenseBlock(in_channels, growth_channels, scale_factor)
        self.l2 = ResidualDenseBlock(in_channels, growth_channels, scale_factor)
        self.l3 = ResidualDenseBlock(in_channels, growth_channels, scale_factor)

        self.scale_factor = scale_factor

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        return out.mul(self.scale_factor) + x

class UpsampleBLock(nn.Module):
    def __init__(self, in_channels):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1)
        self.up_sample = nn.ConvTranspose2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=(2,1), stride=(2,1), padding=0, bias=False)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.up_sample(x)
        x = self.prelu(x)
        return x
