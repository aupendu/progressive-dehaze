import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model():
    return AtmLocal()

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        x = self.conv(x)
        return x

class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownConv, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=7, stride=2, padding=1),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        x = self.mpconv(x)
        return x

class AtmLocal(nn.Module):
    def __init__(self, n_channels=3, out_channels=3):
        super(AtmLocal, self).__init__()
        self.inc = InConv(n_channels, 64) 
        self.down1 = DownConv(64, 128) 
        self.down2 = DownConv(128, 128) 
        self.down3 = DownConv(128, 256) 
        self.down4 = DownConv(256, 256) 
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.dense = nn.Linear(256, out_channels)
    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x) 
        x = self.down2(x) 
        x = self.down3(x) 
        x = self.down4(x) 
        x = self.pool(x)
        x = x.view(-1, 256)
        x = self.dense(x)
        return torch.sigmoid(x)
