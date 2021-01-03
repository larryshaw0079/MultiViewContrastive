"""
@Time    : 2021/1/2 23:14
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : csc.py
@Software: PyCharm
@Desc    : 
"""
import torch.nn as nn
import torchvision

from ..backbone import R1DNet


class CSC(nn.Module):
    def __init__(self, network, input_channels, hidden_channels, feature_dim, pred_steps, use_temperature, temperature,
                 device):
        super(CSC, self).__init__()

        self.time_encoder = R1DNet()
        self.freq_encoder = torchvision.models.resnet18(num_classes=feature_dim)

    def forward(self, x1, x2):
        pass
