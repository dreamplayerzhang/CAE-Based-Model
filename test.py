from networks import Net
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from test_dataset import TestDataset
import torch.nn as nn
from skimage import io
import matplotlib.pyplot as plt

def reconstrution(pieces):
    res = np.ones([600, 800])
    index = 0
    for i in range(75):
        for j in range(100):
            res[i*8:(i+1)*8, j*8:(j+1)*8] = pieces[index, :]
            index += 1

    return res

net = Net()
net.load('./checkpoints/CAE_Nearest2d.pth')
criterion = nn.MSELoss()
testDataset = TestDataset('./dataset/test_patches_csv/8-8.csv')
test_dataloader = DataLoader(testDataset, batch_size=7500, shuffle=False)

for index, item in enumerate(test_dataloader, 0):
    inputs = item.float()
    inputs = Variable(inputs)

    outputs = net(inputs)

    # loss = criterion(outputs, inputs)
    # print(loss.data[0])
    pieces = outputs.data.numpy()
    img = reconstrution(pieces)

    io.imshow(img)
    plt.show()



