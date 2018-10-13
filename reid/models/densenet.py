from __future__ import absolute_import

from torch import nn

from torch.nn import init
from torchvision import models
import torch


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout>0:
            add_block += [nn.Dropout(dropout)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


class DenseNet(nn.Module):

    def __init__(self, pretrained=True, cut_at_pooling=False,
                 num_features=1024, norm=False, dropout=0, num_classes=0 ):
        super(DenseNet,self).__init__()
        model_ft = models.densenet121(pretrained=pretrained)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024
        self.classifier = ClassBlock(num_features, num_classes, dropout=dropout)
        self.norm = norm
        self.cut_at_pooling = cut_at_pooling

    def forward(self, x):
        x = self.model.features(x)
        x = torch.squeeze(x)
        if self.cut_at_pooling:
            return x
        x = self.classifier(x)
        return x

def densenet121(**kwargs):
    return DenseNet(**kwargs)