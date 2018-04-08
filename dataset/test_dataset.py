from torch.utils import data
import pandas as pd
from config import DefaultConfig

opt = DefaultConfig()


class TestDataset(data.Dataset):
    def __init__(self, root):
        df = pd.read_csv(root)
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        df = df.values.reshape([-1, opt.channel, opt.patch_size, opt.patch_size])
        self.df = df

    def __getitem__(self, item):
        res = self.df[item]
        return res

    def __len__(self):
        return len(self.df)
