from mindspore.dataset import context
from load_dataset import create_dataset
from models import EffNetV2, config_s
from mindspore import nn, Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from cfg import config


context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)


if __name__ == '__main__':

    net = EffNetV2(config_S=config_s, num_classes=config.class_num)

    # 构建数据集
    dataset = create_dataset(dataset_path=config.cifar10_path_train,
                             do_train=True,
                             repeat_num=config.epoch_size,
                             batch_size=config.batch_size)
    step_size = dataset.get_dataset_size()
    print(step_size)

    # 损失函数,优化器
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=config.learning_rate, momentum=config.momentum)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # 模型保存
    config_ckpt = CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=10)
    ckpt_cb = ModelCheckpoint(prefix=config.prefix, directory=config.ckpt_files_dir, config=config_ckpt)

    # 输入训练轮次和数据集进行训练
    model = Model(net, loss_fn=loss, optimizer=opt)

    model.train(epoch=config.epoch_size,
                train_dataset=dataset,
                callbacks=[ckpt_cb, TimeMonitor(data_size=step_size), LossMonitor(per_print_times=1)],
                dataset_sink_mode=True)
