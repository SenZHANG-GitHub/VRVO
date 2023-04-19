import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import models
import numpy as np
from torch.autograd import Function
from bilinear_sampler import *

def ssim(x, y):

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = F.avg_pool2d(x, 3, 1)
        mu_y = F.avg_pool2d(y, 3, 1)

        sigma_x = F.avg_pool2d(x**2, 3, 1) - mu_x**2
        sigma_y = F.avg_pool2d(y**2, 3, 1) - mu_y**2
        sigma_xy = F.avg_pool2d(x*y, 3, 1) - mu_x*mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d
        return torch.clamp((1-SSIM)/2, 0, 1)
    
class ReconLoss(nn.Module):
    def __init__(self, alpha=0.85):
        super(ReconLoss, self).__init__()
        self.alpha = alpha

    def forward(self, img0, img1, pred, fb, max_d=80.0):

        x0 = (img0 + 1.0) / 2.0
        x1 = (img1 + 1.0) / 2.0

        assert x0.shape[0] == pred.shape[0]
        assert pred.shape[0] == fb.shape[0]

        new_depth = (pred + 1.0) / 2.0
        new_depth *= max_d
        disp = 1.0 / (new_depth+1e-6)
        tmp = np.array(fb)
        for i in range(new_depth.shape[0]):
            disp[i,:,:,:] *= tmp[i]
            disp[i,:,:,:] /= disp.shape[3] # normlize to [0,1]

        #x0_w = warp(x1, -1.0*disp)
        x0_w = bilinear_sampler_1d_h(x1, -1.0*disp)

        ssim_ = ssim(x0, x0_w)
        l1 = torch.abs(x0-x0_w)
        loss1 = torch.mean(self.alpha * ssim_)
        loss2 = torch.mean((1-self.alpha) * l1)
        loss = loss1 + loss2

        recon_img = x0_w * 2.0-1.0

        return loss, recon_img
