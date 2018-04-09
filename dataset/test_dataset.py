from torch.utils import data
import pandas as pd
from config import DefaultConfig
import numpy as np

opt = DefaultConfig()

class TestDataset(data.Dataset):
    def __init__(self, root, mean_value):
        df = pd.read_csv(root)
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        df = df.values.reshape([-1, opt.channel, opt.patch_size, opt.patch_size])
        self.df = df
        self.mean_image = mean_value

    def __getitem__(self, item):
        res = self.df[item] - self.mean_image
        return res

    def __len__(self):
        return len(self.df)
