import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class Extractor(nn.Module):
    def __init__(self, model='resnet50', pool='max', use_lnorm=True, pretrained=True):
        super().__init__()

        self.base = models.__dict__[model](pretrained=pretrained)
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


class EmbeddingPredictor(nn.Module):
    def __init__(self, bases, embeddings):
        super().__init__()
        self.bases = bases
        self.embeddings = nn.ModuleList(embeddings)

    def forward(self, x):
        """
        Forward propagates each embedding vector.
        """
        fvecs = self.bases(x)
        if not isinstance(fvecs, list) and not isinstance(fvecs, tuple):
            fvecs = [fvecs]

        results = []
        for fvec, emb in zip(fvecs, self.embeddings):
            results.append(emb(fvec))

        return results


class EnsembleExtractor(Extractor):
    def __init__(self, model='resnet50', pool='max', use_lnorm=True, pretrained=True, attention=True):
        super().__init__(model=model, pool=pool, use_lnorm=use_lnorm, pretrained=pretrained)

        self.lnorm1 = None
        self.lnorm2 = None

        if use_lnorm:
            self.lnorm1 = nn.LayerNorm(512, elementwise_affine=False).cuda()
            self.lnorm2 = nn.LayerNorm(1024, elementwise_affine=False).cuda()

        if pool == 'avg':
            self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
            self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            self.pool1 = nn.AdaptiveMaxPool2d((1, 1))
            self.pool2 = nn.AdaptiveMaxPool2d((1, 1))

        #self.top_layer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0) # compression layer
        #self.lat1_layer = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        #self.lat2_layer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        #self.spatial_att1 = nn.Conv2d(1, 1, 7, 7)
        #self.spatial_att2 = nn.Conv2d(1, 1, 7, 7)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        att_src_1 = x
        x = self.base.layer3(x)
        att_src_2 = x
        x = self.base.layer4(x)

        #proj = self.top_layer(x)
        
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)

        if self.lnorm != None:
            x = self.lnorm(x)

        fvec1 = self.pool1(att_src_1)
        fvec1 = fvec1.reshape(fvec1.size(0), -1)
        fvec2 = self.pool2(att_src_2)
        fvec2 = fvec2.reshape(fvec2.size(0), -1)
        if self.lnorm is not None:
            fvec1 = self.lnorm1(fvec1)
            fvec2 = self.lnorm2(fvec2)
        return x, fvec1, fvec2

    def attend_pred(self, x, proj, lat_layer):
        x_proj = lat_layer(x)
        att_blk = torch.cat((x_proj, proj), 1)
        channel_max = torch.max(att_blk, dim=(2, 3))
        channel_mean = torch.mean(att_blk, dim=(2, 3))
