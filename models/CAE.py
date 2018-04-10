import torch.nn as nn
import torch
import time
from config import DefaultConfig

opt = DefaultConfig()

'''
class CAE(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.model_name = str(type(self))

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.ReLU(),
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
'''

'''
class CAE(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.model_name = str(type(self))

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32, 16, 5, padding=2),
            nn.ReLU(),
            # nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(16, 1, 5, padding=2),
            nn.ReLU(),
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
'''

'''
class CAE(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.model_name = str(type(self))

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 45, padding=22),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, 31, padding=15),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 11, padding=5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32, 16, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(16, 1, 5, padding=2),
            nn.ReLU(),
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
'''


class CAE(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.model_name = str(type(self))

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 5, 51, padding=25),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(5, 30, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(30, 40, 5, padding=2),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(40, 30, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(30, 5, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(5, 1, 5, padding=2),
            nn.ReLU(),
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
