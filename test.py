from torch.autograd import Variable
from torch.utils.data import DataLoader
import models
import numpy as np
import os
import skimage.io as io
from dataset import TestDataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def reconstruction(pieces, opt):
    res = np.ones([opt.raw_test_height, opt.raw_test_width])
    step = int(opt.test_patch_stride)
    res = np.pad(res, ((step,), (step,)), mode='edge')
    w_scale = int((opt.raw_test_width - opt.patch_size) / opt.test_patch_stride + 3)
    h_scale = int((opt.raw_test_height - opt.patch_size) / opt.test_patch_stride + 3)

    for i in range(h_scale):
        i_ = i * step
        for j in range(w_scale):
            j_ = j * step
            res[i_:i_ + opt.patch_size, j_:j_ + opt.patch_size] += pieces[i * w_scale + j, 0]

    res = res[step:step+opt.raw_test_height, step:step+opt.raw_test_width]
    res = res/4.0
    return res


def visualize_recon(origin, generated, sub, idx, save_dir):
    plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(221)
    plt.imshow(origin, cmap='gray')
    ax2 = plt.subplot(222)
    plt.imshow(generated, cmap='gray')
    ax3 = plt.subplot(223)
    plt.imshow(np.absolute(sub), cmap='jet')
    ax4 = plt.subplot(224)
    plt.imshow(sub ** 2, cmap='jet')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, str(idx) + '_recon.png')
    plt.savefig(save_path)
    plt.close()


def visualize_patch_jet(residual, idx, save_dir, row, col, patch_size):
    row = int(row / patch_size)
    col = int(col / patch_size)
    mse = np.zeros((row, col))
    for row_idx in range(row):
        for col_idx in range(col):
            patch = residual[row_idx * patch_size:row_idx * patch_size + patch_size,
                    col_idx * patch_size:col_idx * patch_size + patch_size]
            patch = patch ** 2
            patch = np.mean(patch)
            mse[row_idx][col_idx] = patch

    plt.figure(figsize=[24, 12])
    plt.subplots_adjust(top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
    ax1 = plt.subplot(111)
    plt.imshow(mse, cmap='jet')
    plt.colorbar()
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    for i in range(row):
        for j in range(col):
            ax1.text(j, i, '%.3f' % mse[i][j], horizontalalignment='center', verticalalignment='center',
                     fontsize=5)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    jet_path = os.path.join(save_dir, str(idx) + '_residual_jet.png')
    plt.savefig(jet_path)
    plt.close()


def test(opt, load_model_path='default_config'):

    if load_model_path == 'default_config':
        data_info = opt.model + '_pattern_' + opt.pattern_index + '_patch_size_' + str(opt.patch_size) + '_channel_' + str(opt.channel)
        training_parameter = '_batch_size_' + str(opt.train_batch_size) + '_epoch_' + str(opt.max_epoch) + '_lr_' + str(opt.lr)
        checkpoint = opt.load_model_path + data_info + training_parameter + '.pth'
        img_show_path = opt.img_show_path + data_info + training_parameter + '/'
    else:
        img_show_path = opt.img_show_path + load_model_path.replace('.pth', '/')

        index_of_CAE = load_model_path.find('CAE')
        index_of_pattern = load_model_path.find('pattern')
        index_of_patch_size = load_model_path.find('patch_size')
        index_of_channel = load_model_path.find('channel')

        model = load_model_path[index_of_CAE: index_of_pattern - 1]
        pattern_index = load_model_path[index_of_pattern + len('pattern_'): index_of_patch_size - 1]
        patch_size = int(load_model_path[index_of_patch_size + len('patch_size_'): index_of_channel - 1])
        test_patch_stride = int(patch_size / 2)
        new_config = {'model': model,
                      'pattern_index': pattern_index,
                      'patch_size': patch_size,
                      'test_patch_stride': test_patch_stride}
        opt.parse(new_config)

        data_path = opt.train_raw_data_root + opt.pattern_index + '/'
        for item in os.listdir(data_path):
            data_path += item
            break

        img = io.imread(data_path, as_grey=True)
        height = img.shape[0]
        width = img.shape[1]
        new_config = {'raw_train_height': height, 'raw_train_width': width,
                      'raw_test_height': height, 'raw_test_width': width }
        opt.parse(new_config)

        checkpoint = opt.load_model_path + load_model_path

    if os.path.exists(img_show_path):
        pass
    else:
        os.makedirs(img_show_path)

    net = getattr(models, opt.model)()
    net.load(checkpoint)
    if opt.use_gpu:
        net.cuda()

    testDataset = TestDataset(opt=opt)
    w_scale = (opt.raw_test_width - opt.patch_size) / opt.test_patch_stride + 3
    h_scale = (opt.raw_test_height - opt.patch_size) / opt.test_patch_stride + 3
    s = int(w_scale * h_scale)
    test_dataloader = DataLoader(testDataset, batch_size=s, shuffle=False)

    for index, item in enumerate(test_dataloader, 0):
        inputs = item.float()
        original_img = inputs.numpy()
        original_img = reconstruction(original_img, opt)
        inputs = Variable(inputs)
        if opt.use_gpu:
            inputs = inputs.cuda()
        # for i in range(6):
        #     inputs = net(inputs)
        outputs = net(inputs)
        if opt.use_gpu:
            outputs = outputs.cpu()
        outputs = outputs.data.numpy()
        generated_img = reconstruction(outputs, opt)
        residual_img = generated_img - original_img

        visualize_recon(original_img, generated_img, residual_img, index, img_show_path)
        visualize_patch_jet(residual_img, index, img_show_path, opt.raw_test_height, opt.raw_test_width, opt.heatmap_patch_size)
