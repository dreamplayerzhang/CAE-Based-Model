import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from train import train
from test import test
from config import DefaultConfig

opt = DefaultConfig()

train(opt, initialize=True)
test(opt)


# for item in os.listdir(opt.load_model_path):
#     test(opt, load_model_path=item)
