from torch.utils import data
import os
from skimage import io


class TextileData(data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]

    def __getitem__(self, item):
        img_path = self.imgs[item]
        img = io.imread(img_path, as_grey=True)
        return img

    def __len__(self):
        return len(self.imgs)
