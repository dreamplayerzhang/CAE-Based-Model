import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

dtype = torch.FloatTensor
batch_size = 4


class Net(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, 1, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, 1, padding=0),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1, padding=0),
            nn.ReLU(),
            # nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 16, 1, padding=0),
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

net = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

data = pd.read_csv('./dataset/patches_csv/8-4.csv')
data.drop(['Unnamed: 0'], axis=1, inplace=True)
data = data.sample(frac=0.1)
data = data.values.reshape([-1, 1, 8, 8])
batches = int(data.shape[0])
# batches = int(data.shape[0] / batch_size)

for epoch in range(2):
    running_loss = 0.0
    for i in range(batches):
        inputs = data[i].reshape([1, 1, 8, 8])
        # inputs = data[i:i + batch_size, :, :, :]
        inputs = torch.from_numpy(inputs).float()
        inputs = Variable(inputs)

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, inputs)
        loss.backward()

        optimizer.step()

        running_loss += loss.data[0]
        if i == 0:
            print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, running_loss))
        if i % 500 == 499:
            print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, running_loss / 500))
            # print(inputs)
            # print(outputs)
            torch.save(net.state_dict(), 'CAE_Nearest2d.pth')
            running_loss = 0.0



