import torch
from torch import nn
import pytorch_msssim
import pytorch_ssim
import kornia
from torchvision.models.vgg import vgg16


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        # vgg = vgg16(pretrained=True)
        # loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        # for param in loss_network.parameters():
        #     param.requires_grad = False
        # self.loss_network = loss_network

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.ssim = pytorch_ssim.SSIM()
        self.msssim = pytorch_msssim.MSSSIM()
        self.tv_loss = TVLoss()


    def forward(self, output, target):
        # Structural Loss
        s_loss = 0.7*(1 - self.msssim(output, target, normalize=True)) / 2 + 0.3*(1 - self.ssim(output, target)) / 2
        # p_output = torch.ones([output.size()[0], 3, output.size()[2], output.size()[3]], device=self.device)*output
        # p_target = torch.ones([output.size()[0], 3, output.size()[2], output.size()[3]], device=self.device)*target
        # perception_loss = self.mse(self.loss_network(p_output), self.loss_network(p_target))
        # Image Loss
        mse_loss = self.mse(output, target)
        l1_loss = self.l1(output, target)
        grad_loss = gradient_loss(output, target)
        tv_loss = self.tv_loss(output)

        return mse_loss, l1_loss, s_loss, grad_loss, tv_loss

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

def gradient_loss(output, target, alpha=1):
    output_grad = kornia.spatial_gradient(output)
    output_gx = output_grad[:, :, 0, :]
    output_gy = output_grad[:, :, 1, :]

    target_grad = kornia.spatial_gradient(target)
    target_gx = target_grad[:, :, 0, :]
    target_gy = target_grad[:, :, 1, :]

    grad_diff_x = torch.abs(output_gx - target_gx)
    grad_diff_y = torch.abs(output_gy - target_gy)

    return torch.mean(grad_diff_x ** alpha + grad_diff_y ** alpha)