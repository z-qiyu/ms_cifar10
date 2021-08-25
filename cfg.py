
class config:
    learning_rate = 1e-3
    batch_size = 1
    epoch_size = 5
    class_num = 10
    momentum = 0.9
    cifar10_path_train = r".\dataset\train"
    cifar10_path_eval = r'.\dataset\test'
    ckpt_files_dir = r'.\ckpt'
    prefix = 'EffNet_V2'
    device_target = "CPU"
