from torch import nn


def conv_1x1_bn(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(c_out),
        nn.SiLU(),
    )


def conv_nxn_bn(c_in, c_out, kernel_size, stride, groups=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size, stride,
                  padding=(kernel_size // 2), groups=groups, bias=False),
        nn.BatchNorm2d(c_out),
        nn.SiLU(),
    )
