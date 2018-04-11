import models
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import TestDataset
from skimage import io
import matplotlib.pyplot as plt
from config import DefaultConfig

opt = DefaultConfig()


def reconstruction(pieces):
    res = np.ones([opt.raw_test_size, opt.raw_test_size])
    scope = int(opt.raw_test_size / opt.patch_size)
    for i in range(scope):
        for j in range(scope):
            res[i*opt.patch_size:(i+1)*opt.patch_size, j*opt.patch_size:(j+1)*opt.patch_size] = pieces[i*scope+j, :]

    return res


def test(mean_value):
    net = getattr(models, opt.model)()
    net.load(opt.load_model_path + str(opt.patch_size) + 'Patches.pth')
    if opt.use_gpu:
        net.cuda()
    testDataset = TestDataset(opt.test_patches_root + str(opt.patch_size) + '-' + str(opt.patch_size) + '.csv', mean_value=mean_value)
    s = int(opt.raw_test_size / opt.patch_size)
    test_dataloader = DataLoader(testDataset, batch_size=s * s, shuffle=False)

    for index, item in enumerate(test_dataloader, 0):
        inputs = item.float()
        raw_img = inputs.numpy()
        raw_img = reconstruction(raw_img)
        inputs = Variable(inputs)
        if opt.use_gpu:
            inputs = inputs.cuda()
        outputs = net(inputs)
        if opt.use_gpu:
            outputs = outputs.cpu()
        outputs = outputs.data.numpy()
        outputs = outputs + mean_value
        res_img = reconstruction(outputs)
        residual_img = res_img - raw_img
        square_residual_img = residual_img * residual_img

        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax1.imshow(raw_img)

        ax2 = fig.add_subplot(222)
        ax2.imshow(res_img)

        ax3 = fig.add_subplot(223)
        ax3.imshow(residual_img)

        ax4 = fig.add_subplot(224)
        ax4.imshow(square_residual_img)
        # plt.show()
        fig.savefig(opt.img_show_path + str(opt.patch_size) + '/' + opt.model_index + str(index) + '.png')
