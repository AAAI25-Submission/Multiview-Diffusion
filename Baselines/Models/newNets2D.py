import torch.nn as nn
from efficientnet_pytorch import EfficientNet as efficientnet
from torchvision import models
import torch

def EfficientNetB4():
    model = efficientnet.from_name("efficientnet-b4")

    # 获取原始第一层卷积的权重
    first_conv_layer = model._conv_stem
    original_weight = first_conv_layer.weight

    # 创建一个新的第一层卷积，其输入通道数为1
    # 注意保持其他参数（如输出通道数、内核大小等）与原始卷积相同
    new_first_conv_layer = nn.Conv2d(1, first_conv_layer.out_channels,
                                     kernel_size=first_conv_layer.kernel_size,
                                     stride=first_conv_layer.stride,
                                     padding=first_conv_layer.padding,
                                     bias=first_conv_layer.bias)

    # 将原始权重的平均值分配给新层（从3通道到1通道）
    # 这里我们沿着通道维度取平均值
    new_weight = original_weight.mean(dim=1, keepdim=True)

    # 更新新层的权重
    new_first_conv_layer.weight = nn.Parameter(new_weight)

    # 用新的第一层卷积替换模型中的原始第一层卷积
    model._conv_stem = new_first_conv_layer

    # model._fc = nn.Linear(model._fc.in_features, num_classes)
    return model

class downStreamClassifier(nn.Module):
    def __init__(self, baseModel, num_classes):
        super(downStreamClassifier, self).__init__()
        self.baseModel = baseModel
        self.baseModel._fc = nn.Linear(self.baseModel._fc.in_features, num_classes)

    def forward(self, x):
        x = self.baseModel(x)
        return x