class DefaultConfig(object):
    model = 'CAE1'       # 使用模型，名字与models/__init__.py中的名字一致
    pattern_index = '5'

    train_raw_data_root = './dataset/pattern/pattern'    # 训练原始数据集（正样本）的存放路径
    test_raw_data_root = './dataset/bad_pattern/bad_pattern'   # 测试原始数据集（负样本）的存放路径
    train_patches_root = './dataset/train_patches_csv/pattern'     # 提取后的训练数据集（正样本patches）存放路径
    test_patches_root = './dataset/test_patches_csv/pattern'        # 提取后的测试数据集（正样本patches）存放路径
    load_model_path = './checkpoints/'      # 加载、保存训练模型的路径
    img_show_path = './img_show/'
    log_dir = './logs/'

    raw_train_height = 1200      # 原始训练数据（正样本）的图像高度
    raw_train_width = 1600       # 原始训练数据（正样本）的图像宽度
    raw_test_height = 1200       # 原始测试数据（负样本）的图像高度
    raw_test_width = 1600        # 原始测试数据（负样本）的图像高度
    patch_size = 400
    train_patch_stride = 100
    test_patch_stride = 200
    channel = 1
    heatmap_patch_size = 10
    use_gpu = True
    num_workers = 4
    train_batch_size = 16
    print_freq = 20
    max_epoch = 200
    lr = 0.0001
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
