from skimage import io, transform
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

path = './dataset/pattern'
ls = os.listdir(path)
N, W, H, C = len(ls), 800, 600, 1
patch_size, stride = 32, 8
raw_data = np.zeros([N, C, H, W])
index = 0

for filename in ls:
    img = io.imread(path+'/'+filename, as_grey=True)
    raw_data[index] = img
    index += 1

w_scale = (W - patch_size)/stride + 1
h_scale = (H - patch_size)/stride + 1
total_patches = w_scale * h_scale * N
total_patches = int(total_patches)
print(total_patches)

patches = np.zeros([total_patches, C, patch_size, patch_size])
index = 0

for i in range(int(h_scale)):
    for j in range(int(w_scale)):
        section = raw_data[:, :, i*stride:i*stride + patch_size, j*stride:j*stride + patch_size]
        patches[index*15:(index+1)*15] = section
        index += 1

print(patches.shape)
#
# patches_ = np.reshape(patches, (total_patches, -1))
# df = pd.DataFrame(patches_)
# df.to_csv('./dataset/patches_csv/' + str(patch_size) + '-' + str(stride) + '.csv', index=True)
