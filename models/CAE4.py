import torch.nn as nn
import torch
import time
from config import DefaultConfig

opt = DefaultConfig()


class CAE4(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.model_name = str(type(self))

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 100, 50, stride=25),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(100, 1, 50, stride=25),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        torch.save(self.state_dict(), name)
