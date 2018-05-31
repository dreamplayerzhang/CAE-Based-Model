import torch.nn as nn
import torch.nn.functional as F
import torch
import time
from config import DefaultConfig

opt = DefaultConfig()


class CAE6_2(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.model_name = str(type(self))

        self.encoder_conv1 = nn.Conv2d(1, 10, kernel_size=6, padding=0, stride=1)
        self.encoder_conv2 = nn.Conv2d(10, 20, kernel_size=10, padding=0, stride=1)
        self.encoder_conv3 = nn.Conv2d(20, 40, kernel_size=11, padding=0, stride=1)
        self.encoder_conv4 = nn.Conv2d(40, 60, kernel_size=11, padding=0, stride=1)

        self.decoder_deconv4= nn.ConvTranspose2d(60, 40, kernel_size=11, padding=0, stride=1)
        self.decoder_deconv3 = nn.ConvTranspose2d(40, 20, kernel_size=11, padding=0, stride=1)
        self.decoder_deconv2 = nn.ConvTranspose2d(20, 10, kernel_size=10, padding=0, stride=1)
        self.decoder_deconv1 = nn.ConvTranspose2d(10, 1, kernel_size=6, padding=0, stride=1)

    def forward(self, x):
        x, indices1 = F.max_pool2d(F.relu(self.encoder_conv1(x)), kernel_size=3, stride=3, return_indices=True)
        x, indices2 = F.max_pool2d(F.relu(self.encoder_conv2(x)), kernel_size=2, stride=2, return_indices=True)
        x = F.relu(self.encoder_conv3(x))
        x = F.relu(self.encoder_conv4(x))

        x = F.relu(self.decoder_deconv4(x))
        x = F.max_unpool2d(F.relu(self.decoder_deconv3(x)), indices=indices2, kernel_size=2, stride=2)
        x = F.max_unpool2d(F.relu(self.decoder_deconv2(x)), indices=indices1, kernel_size=3, stride=3)
        x = F.relu(self.decoder_deconv1(x))

        return x

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        torch.save(self.state_dict(), name)
