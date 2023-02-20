import math
import torch
from torch import nn

'''Ref. ESRGAN'''

class Generator(nn.Module):
    def __init__(self, n_basic_block=8):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
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
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x = self.baskc_block(x1)
        x = self.layer2(x)
        x = self.layer3(x+x1)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, image_size: int = 128) -> None:
        super(Discriminator, self).__init__()

        feature_size = int(image_size // 32)

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # input is 256
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        # 20992 : 1300 full
        # 10752 : 650 crop
        self.classifier = nn.Sequential(
            nn.Linear(512*feature_size*feature_size, 100),
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