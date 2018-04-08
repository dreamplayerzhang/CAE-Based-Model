class DefaultConfig(object):
    env = 'default'
    model = 'CAE'
    train_raw_data_root = './dataset/pattern/'
    test_raw_data_root = './dataset/bad_pattern/'
    train_patches_root = './dataset/train_patches_csv/'
    test_patches_root = './dataset/test_patches_csv/'
    load_model_path = './checkpoints/'
    load_model = './checkpoints/CAE_Nearest2d.pth'
    raw_train_height = 600
    raw_train_width = 800
    raw_test_size = 512
    patch_size = 32
    patch_stride = 4
    channel = 1
    use_gpu = True
    num_workers = 4
    train_batch_size = 5
    print_freq = 20
    max_epoch = 5
    lr = 0.01
    momentum = 0.9

    def parse(self, dicts):
        for k, v in dicts.items():
            if hasattr(self, k):
                setattr(self, k, v)
