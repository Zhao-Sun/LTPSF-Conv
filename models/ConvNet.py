import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from layers.Invertible import RevIN


class series_decomp_conv(nn.Module):

    def __init__(self, ks=None, ch_in=None):
        super(series_decomp_conv, self).__init__()
        self.conv = nn.Conv1d(ch_in, ch_in, groups=ch_in, kernel_size=ks, padding='same')


    def forward(self, x):

        moving_mean = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        return moving_mean


class ConvNet(nn.Module):

    def __init__(self, ks=None, ch_in=None):
        super(ConvNet, self).__init__()

        self.decompsition = series_decomp_conv(ks, ch_in)


    def forward(self, x):

        output_init = self.decompsition(x)

        return output_init


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.kernel_size = configs.kernel_size
        self.individual = configs.individual
        self.channels = configs.channel
        self.reduction = configs.reduction
        self.r = configs.r
        self.individual = configs.individual
        self.ConvNet = ConvNet(self.kernel_size, self.channels)

        self.rev = RevIN(configs.channel) if configs.rev else None

        self.Linear = nn.ModuleList([
            nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)
        ]) if self.individual else nn.Linear(self.seq_len, self.pred_len)

    def forward_loss(self, pred, true):
        return F.mse_loss(pred, true)

    def forward(self, x, y):

        x = self.rev(x, 'norm') if self.rev else x

        seq_last = torch.mean(x, dim=1, keepdim=True)
        x = x - seq_last
        x = self.ConvNet(x)

        if self.individual:
            pred = torch.zeros_like(y)
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x[:, :, idx])
        else:
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)


        pred = pred + seq_last

        pred = self.rev(pred, 'denorm') if self.rev else pred

        return pred, self.forward_loss(pred, y)
