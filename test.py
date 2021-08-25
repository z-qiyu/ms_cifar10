from mindspore import Model, nn
from mindspore.nn import Accuracy
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from load_dataset import create_dataset
from models import EffNetV2, config_s
from cfg import config
import os
from mindspore.dataset import context

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)


def test(ckpt_file_name: str):
    net = EffNetV2(config_s, num_classes=config.class_num)
    net.set_train(False)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=config.learning_rate, momentum=config.momentum)

    param_dict = load_checkpoint(ckpt_file_name)
    # 导入模型参数
    load_param_into_net(net, param_dict)

    # 构建数据集
    dataset = create_dataset(dataset_path=config.cifar10_path_eval, do_train=False, batch_size=config.batch_size)

    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'Accuracy': Accuracy()})
    res = model.eval(dataset)
    print("result:", res, "ckpt=", ckpt_file_name)


for i in os.listdir(config.ckpt_files_dir):
    file_name = os.path.join(config.ckpt_files_dir, i)
    if file_name[-4:] == 'ckpt':
        test(file_name)
    else:
        continue
