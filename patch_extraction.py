import pandas as pd
from rawdata_preprocess import TextileData
import numpy as np

def patches_generation(root='./dataset/pattern/', patch_size=8, stride=4, mode='train', save2csv=False):
    if mode == 'test':
        stride = patch_size

    raw_dataset = TextileData(root)
    N, W, H, C = len(raw_dataset), 800, 600, 1
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
            path = './dataset/train_patches_csv/'
        elif mode == 'test':
            path = './dataset/test_patches_csv/'

        df.to_csv(path + str(patch_size) + '-' + str(stride) + '.csv')




