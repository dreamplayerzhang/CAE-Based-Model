from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TrainDataset
import models
from config import DefaultConfig
from torch.nn import init


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight)
        init.constant(m.bias, 0.0)


def train():
    opt = DefaultConfig()
    net = getattr(models, opt.model)()
    net.apply(weights_init)
    # net.load(opt.load_model_path + str(opt.patch_size) + 'Patches.pth')
    if opt.use_gpu:
        net.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum)
    trainDataset = TrainDataset(opt.train_patches_root + str(opt.patch_size) + '-' + str(opt.patch_stride) + '.csv')
    train_dataloader = DataLoader(trainDataset, batch_size=opt.train_batch_size, shuffle=True)
    # mean_value = trainDataset.mean_image

    for epoch in range(opt.max_epoch):
        running_loss = 0.0
        for index, item in enumerate(train_dataloader, 1):
            inputs = item.float()
            inputs = Variable(inputs)
            if opt.use_gpu:
                inputs = inputs.cuda()

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if index == 1:
                print('[%d, %5d] loss: %.6f' % (epoch + 1, index, running_loss))
            if index % opt.print_freq == 0:
                print('[%d, %5d] loss: %.9f' % (epoch + 1, index, running_loss / opt.print_freq))
                net.save(opt.load_model_path + str(opt.patch_size) + 'Patches.pth')
                running_loss = 0.0

    # return mean_value