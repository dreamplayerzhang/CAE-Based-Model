from torch.utils import data
import pandas as pd


class TrainDataset(data.Dataset):
    def __init__(self, root):
        df = pd.read_csv(root)
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        df = df.sample(frac=0.001)
        df = df.values.reshape([-1, 1, 8, 8])
        self.df = df

    def __getitem__(self, item):
        res = self.df[item]
        return res

    def __len__(self):
        return len(self.df)



