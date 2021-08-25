"""
create train or eval dataset.
"""
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2


def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32

    Returns:
        dataset
    """

    ds = de.Cifar10Dataset(dataset_path, num_parallel_workers=4, shuffle=True)

    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    random_crop_op = C.RandomCrop((32, 32), (4, 4, 4, 4))
    random_horizontal_flip_op = C.RandomHorizontalFlip(prob=0.5)

    resize_op = C.Resize((resize_height, resize_width))
    rescale_op = C.Rescale(rescale, shift)
    normalize_op = C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

    change_swap_op = C.HWC2CHW()

    trans = []
    if do_train:
        trans += [random_crop_op, random_horizontal_flip_op]

    trans += [resize_op, rescale_op, normalize_op, change_swap_op]

    type_cast_op = C2.TypeCast(mstype.int32)

    ds = ds.map(input_columns="label", operations=type_cast_op)
    ds = ds.map(input_columns="image", operations=trans)

    ds = ds.shuffle(buffer_size=100)

    ds = ds.batch(batch_size, drop_remainder=True)

    ds = ds.repeat(repeat_num)

    return ds

