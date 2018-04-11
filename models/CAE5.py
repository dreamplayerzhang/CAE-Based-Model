import torch.nn as nn
import torch
import time
from config import DefaultConfig

opt = DefaultConfig()


class CAE5(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.model_name = str(type(self))

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 20, 64, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 40, 13, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(40, 60, 3, stride=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(60, 40, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(40, 20, 13, stride=4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(20, 1, 64, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = opt.load_model_path + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
