import mindspore.nn as nn
import math
from mindspore.common.initializer import Normal


# mindspore实现AdaptiveAvgPool2d(1)
def AdaptiveAvgPool2d1x1(x):
    pool = nn.AvgPool2d((x.shape[2], x.shape[3]))
    return pool(x)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU激活函数
class SiLU(nn.Cell):
    def __init__(self, *args, **kwargs):
        super(SiLU, self).__init__(*args, **kwargs)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        return x * self.sigmoid(x)


# SE注意力
class SELayer(nn.Cell):
    def __init__(self, inp, oup, reduction=4):
        super().__init__()
        self.fc = nn.SequentialCell(
            nn.Dense(oup, _make_divisible(inp // reduction, 8), weight_init=Normal(0.001), bias_init=Normal(0)),
            SiLU(),
            nn.Dense(_make_divisible(inp // reduction, 8), oup, weight_init=Normal(0.001), bias_init=Normal(0)),
            nn.Sigmoid()
        )

    def construct(self, x):
        b, c, _, _ = x.shape
        y = AdaptiveAvgPool2d1x1(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# 第一层3x3卷积
def conv_3x3_bn(inp, oup, stride):
    return nn.SequentialCell(
        nn.Conv2d(inp, oup, 3, stride, padding=1, pad_mode='pad',
                  weight_init=Normal(math.sqrt(2. / (9 * oup)))),
        nn.BatchNorm2d(oup, gamma_init=Normal(1), beta_init=Normal(0)),
        SiLU()
    )


# 最后一层1x1卷积
def conv_1x1_bn(inp, oup):
    return nn.SequentialCell(
        nn.Conv2d(inp, oup, 1, 1, padding=0, weight_init=Normal(math.sqrt(2. / oup))),
        nn.BatchNorm2d(oup, gamma_init=Normal(1), beta_init=Normal(0)),
        SiLU()
    )


class MBConv(nn.Cell):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.SequentialCell(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, padding=0,
                          weight_init=Normal(math.sqrt(2. / hidden_dim))),
                nn.BatchNorm2d(hidden_dim, gamma_init=Normal(1), beta_init=Normal(0)),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding=1, group=hidden_dim,
                          pad_mode='pad',
                          weight_init=Normal(math.sqrt(2. / (9 * hidden_dim)))),
                nn.BatchNorm2d(hidden_dim, gamma_init=Normal(1), beta_init=Normal(0)),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, padding=0,
                          weight_init=Normal(math.sqrt(2. / oup))),
                nn.BatchNorm2d(oup, gamma_init=Normal(1), beta_init=Normal(0)),
            )
        else:
            self.conv = nn.SequentialCell(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, padding=1, pad_mode='pad',
                          weight_init=Normal(math.sqrt(2. / (9 * hidden_dim)))),
                nn.BatchNorm2d(hidden_dim, gamma_init=Normal(1), beta_init=Normal(0)),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, padding=0,
                          weight_init=Normal(math.sqrt(2. / oup))),
                nn.BatchNorm2d(oup, gamma_init=Normal(1), beta_init=Normal(0)),
            )

    def construct(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Cell):
    def __init__(self, config_S, num_classes=10, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.config_S = config_S

        # 第一层
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # 构造中间层
        block = MBConv
        for t, c, n, s, use_se in self.config_S:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.SequentialCell(*layers)
        # 最后一层
        output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.classifier = nn.Dense(output_channel, num_classes, weight_init=Normal(0.001), bias_init=Normal(0))

    def construct(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = AdaptiveAvgPool2d1x1(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


config_s = [
    # t, c, n, s, SE
    [1, 24, 2, 1, 0],
    [4, 48, 4, 2, 0],
    [4, 64, 4, 2, 0],
    [4, 128, 6, 2, 1],
    [6, 160, 9, 1, 1],
    [6, 256, 15, 2, 1],
]
