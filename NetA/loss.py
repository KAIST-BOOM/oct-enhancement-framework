import torch
from torch import nn
import kornia

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, fake, real):
        # Loss
        mse_loss = self.mse(fake, real)
        l1_loss = self.l1(fake, real)
        grad_loss = gradient_loss(fake, real)

        return mse_loss, l1_loss, grad_loss

def gradient_loss(output, target, alpha=1):
    output_grad = kornia.spatial_gradient(output)
    output_gx = output_grad[:, :, 0, :]
    output_gy = output_grad[:, :, 1, :]

    target_grad = kornia.spatial_gradient(target)
    target_gx = target_grad[:, :, 0, :]
    target_gy = target_grad[:, :, 1, :]

    grad_diff_x = torch.abs(output_gx - target_gx)
    grad_diff_y = torch.abs(output_gy - target_gy)

    return torch.mean(grad_diff_y ** alpha)