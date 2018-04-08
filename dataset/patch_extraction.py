import pandas as pd
from dataset import TextileData
import numpy as np
from config import DefaultConfig

opt = DefaultConfig()


def patches_generation(root='', patch_size=8, stride=4, mode='train', save2csv=False):
    if mode == 'train':
        root = opt.train_raw_data_root
        patch_size = opt.patch_size
        stride = opt.patch_stride
    elif mode == 'test':
        root = opt.test_raw_data_root
        patch_size = opt.patch_size
        stride = patch_size

    raw_dataset = TextileData(root)
    N, W, H, C = len(raw_dataset), 0, 0, 0
    if mode == 'train':
        W = opt.raw_train_width
        H = opt.raw_train_height
        C = opt.channel
    elif mode == 'test':
        W = opt.raw_test_size
        H = opt.raw_test_size
        C = opt.channel

    raw_data = np.ones([N, C, H, W])

    for index, item in enumerate(raw_dataset):
        raw_data[index] = item

    w_scale = (W - patch_size)/stride + 1
    h_scale = (H - patch_size)/stride + 1
    total_patches = w_scale * h_scale * N
    total_patches = int(total_patches)
    print(total_patches)

    patches = np.ones([total_patches, C, patch_size, patch_size])
    index = 0

    if mode == 'train':
        for i in range(int(h_scale)):
            for j in range(int(w_scale)):
                section = raw_data[:, :, i*stride:i*stride + patch_size, j*stride:j*stride + patch_size]
                patches[index*15:(index+1)*15] = section
                index += 1
    elif mode == 'test':
        pass
        for n in range(N):
            for i in range(int(h_scale)):
                for j in range(int(w_scale)):
                    section = raw_data[n, :, i*stride:i*stride + patch_size, j*stride:j*stride + patch_size]
                    patches[index] = section
                    index += 1

    print(patches.shape)

    if save2csv:
        patches_ = np.reshape(patches, (total_patches, -1))
        df = pd.DataFrame(patches_)
        if mode == 'train':
            path = opt.train_patches_root
        elif mode == 'test':
            path = opt.test_patches_root

        df.to_csv(path + str(patch_size) + '-' + str(stride) + '.csv')
