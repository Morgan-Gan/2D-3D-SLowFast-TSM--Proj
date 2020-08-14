import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Hidden(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(Hidden, self).__init__()

    def forward(self, x):
        out=x.view(x.shape[0],-1)
        out = out.view(-1, out.size(1))
        return out

def weight_init(m):
    # 也可以判断是否为conv2d，使用相应的初始化方式
    if isinstance(m, nn.Conv3d):
        print("using kaiming")
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

def hidden50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = Hidden(2304,2304,2)
    return model