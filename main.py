from train import train
from test import test
from dataset import TrainDataset
from config import DefaultConfig

# opt = DefaultConfig()


# trainDataset = TrainDataset(opt.train_patches_root + str(opt.patch_size) + '-' + str(opt.patch_stride) + '.csv')
mean = train()
test(mean)


"""
from dataset import patches_generation

patches_generation(mode='test', save2csv=True)
"""