import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class Extractor(nn.Module):
    def __init__(self, model='resnet50', pool='max', use_lnorm=True):
        super().__init__()

        self.base = models.__dict__[model](pretrained=True)
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, 1))

        self.lnorm = None
        if use_lnorm:
            self.lnorm = nn.LayerNorm(2048, elementwise_affine=False).cuda()

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        x = self.pool(x)
        x = x.reshape(x.size(0), -1)

        if self.lnorm != None:
            x = self.lnorm(x)

        return x
