import torch.nn as nn
import torch.nn.functional as F
import torch
import time
from config import DefaultConfig

opt = DefaultConfig()


class CAE6_1(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.model_name = str(type(self))

        self.encoder_conv1 = nn.Conv2d(1, 20, kernel_size=6, padding=0, stride=1)
        self.maxpooling1 = nn.MaxPool2d(3, 3)
        self.encoder_conv2 = nn.Conv2d(20, 40, kernel_size=10, padding=0, stride=1)
        self.maxpooling2 = nn.MaxPool2d(2, 2)
        self.encoder_conv3 = nn.Conv2d(40, 60, kernel_size=11, padding=0, stride=1)
        self.encoder_conv4 = nn.Conv2d(60, 80, kernel_size=11, padding=0, stride=1)

        self.decoder_deconv4= nn.ConvTranspose2d(80, 60, kernel_size=11, padding=0, stride=1)
        self.decoder_deconv3 = nn.ConvTranspose2d(60, 40, kernel_size=11, padding=0, stride=1)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_deconv2 = nn.ConvTranspose2d(40, 20, kernel_size=10, padding=0, stride=1)
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=3)
        self.decoder_deconv1 = nn.ConvTranspose2d(20, 1, kernel_size=6, padding=0, stride=1)

    def forward(self, x):
        x = F.relu(self.encoder_conv1(x))
        x = self.maxpooling1(x)
        x = F.relu(self.encoder_conv2(x))
        x = self.maxpooling2(x)
        x = F.relu(self.encoder_conv3(x))
        x = F.relu(self.encoder_conv4(x))

        x = F.relu(self.decoder_deconv4(x))
        x = F.relu(self.decoder_deconv3(x))
        x = self.upsample2(x)
        x = F.relu(self.decoder_deconv2(x))
        x = self.upsample1(x)
        x = F.relu(self.decoder_deconv1(x))

        return x

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        torch.save(self.state_dict(), name)
