class DefaultConfig(object):
    env = 'default'     # visdom 环境
    model = 'CAE5'       # 使用模型，名字与models/__init__.py中的名字一致
    model_index = '005/'
    pattern_index = '3/'

    train_raw_data_root = './dataset/pattern/pattern' + pattern_index    # 训练原始数据集（正样本）的存放路径
    test_raw_data_root = './dataset/bad_pattern/bad_pattern' + pattern_index  # 测试原始数据集（负样本）的存放路径
    train_patches_root = './dataset/train_patches_csv/pattern' + pattern_index     # 提取后的训练数据集（正样本patches）存放路径
    test_patches_root = './dataset/test_patches_csv/pattern' + pattern_index       # 提取后的测试数据集（正样本patches）存放路径
    load_model_path = './checkpoints/pattern' + pattern_index + model_index      # 加载、保存训练模型的路径
    img_show_path = './img_show/pattern' + pattern_index

    raw_train_height = 600      # 训练原始数据（正样本）的图像高度
    raw_train_width = 800       # 训练原始数据（正样本）的图像宽度
    raw_test_size = 512         # 测试原始数据（负样本）的图像高度（宽=高）
    patch_size = 128
    patch_stride = 8
    channel = 1
    mean_value = 0
    use_gpu = True
    num_workers = 4
    train_batch_size = 5
    print_freq = 100
    max_epoch = 50
    lr = 0.01
    lr_decay = 0.95
    weight_decay = 1e-4
    momentum = 0.9

    def parse(self, dicts):
        """
        根据传入字典dicts更新config参数
        :param dicts:
        :return: None
        """
        for k, v in dicts.items():
            if hasattr(self, k):
                setattr(self, k, v)
