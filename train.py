from networks import Net
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from train_dataset import TrainDataset

net = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
trainDataset = TrainDataset('./dataset/train_patches_csv/8-4.csv')
train_dataloader = DataLoader(trainDataset, batch_size=5, shuffle=True)

for epoch in range(2):
    running_loss = 0.0
    for index, item in enumerate(train_dataloader, 0):
        inputs = item.float()
        inputs = Variable(inputs)

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, inputs)
        loss.backward()

        optimizer.step()

        running_loss += loss.data[0]
        if index == 0:
            print('[%d, %5d] loss: %.6f' % (epoch + 1, index + 1, running_loss))
        if index % 20 == 19:
            print('[%d, %5d] loss: %.6f' % (epoch + 1, index + 1, running_loss / 20))
            # torch.save(net.state_dict(), 'CAE_Nearest2d.pth')
            running_loss = 0.0
