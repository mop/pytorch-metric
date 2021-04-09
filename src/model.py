import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import copy


class SwitchableBatchNorm(nn.Module):
    def __init__(self, batch_norm_layer, num_batchnorms=2):
        super().__init__()
        self.active = 0
        self.num_batchnorms = num_batchnorms

        self.batch_norm_layers = []
        for i in range(self.num_batchnorms):
            self.batch_norm_layers.append(copy.deepcopy(batch_norm_layer))
        self.batch_norm_layers = nn.ModuleList(self.batch_norm_layers)

    def forward(self, x):
        return self.batch_norm_layers[self.active](x)

    @classmethod
    def convert_switchable_batchnorm(cls, module, num_batchnorms=2):
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = SwitchableBatchNorm(module, num_batchnorms=num_batchnorms)
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_switchable_batchnorm(child, num_batchnorms))
        del module
        return module_output

    @classmethod
    def switch_to(cls, module, active=0):
        if isinstance(module, SwitchableBatchNorm):
            module.active = active
        for name, child in module.named_children():
            cls.switch_to(child, active)


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



class AttentionExtractor(nn.Module):
    def __init__(self, model='resnet50', pool='avg', use_lnorm=True, pretrained=True):
        super().__init__()

        self.attention = CBAMAttention()

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

        x = self.attention(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)

        if self.lnorm != None:
            x = self.lnorm(x)

        return x


class CBAMAttention(nn.Module):
    def __init__(self, channels_in=2048, rate=4):
        super().__init__()
        self.channels_in = channels_in
        self.rate = rate

        self.channel_attention1 = nn.Linear(self.channels_in, self.channels_in // self.rate)
        self.channel_relu = nn.ReLU()
        self.channel_attention2 = nn.Linear(self.channels_in // self.rate, self.channels_in)

        self.channel_spatial = nn.Conv2d(2, 1, 7, padding=(3, 3))

    def forward(self, x):
        red_mean = torch.mean(x, dim=(2,3))
        red_max = torch.amax(x, dim=(2,3))
        exc1 = self.channel_attention2(self.channel_relu(self.channel_attention1(red_mean)))
        exc2 = self.channel_attention2(self.channel_relu(self.channel_attention1(red_max)))
        exc = torch.sigmoid(exc1 + exc2)

        att1 = exc[:, :, None, None] * x

        feat1 = torch.mean(att1, dim=1, keepdim=True)
        feat2 = torch.amax(att1, dim=1, keepdim=True)
        feat = torch.cat((feat1, feat2), axis=1)

        att_chan = torch.sigmoid(self.channel_spatial(feat))
        return att1 * att_chan
        


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


class SimSiamEmbeddingPredictor(EmbeddingPredictor):
    def __init__(self, bases, embeddings, sim_siam):
        super().__init__(bases, embeddings)
        self.sim_siam = sim_siam

    def forward(self, x):
        fvecs = self.bases(x)

        if not isinstance(fvecs, list) and not isinstance(fvecs, tuple):
            fvecs = [fvecs]

        results = []
        for fvec, emb in zip(fvecs, self.embeddings):
            results.append(emb(fvec))

        return results + list(self.sim_siam(fvecs[0]))
        


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


class ProjectionMLP(nn.Module):
    """
    Projetion layer from SimSiam
    """
    def __init__(self, in_dim, mid_dim, out_dim):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l3 = nn.Sequential(
            nn.Linear(mid_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


class PredictionMLP(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        return self.l2(self.l1(x))


class SimSiamEmbedding(nn.Module):
    def __init__(self, in_dim=2048, out_dim=2048):
        super().__init__()
        self.projection = ProjectionMLP(in_dim, in_dim, out_dim)
        self.prediction = PredictionMLP(in_dim, in_dim//4, out_dim)

    def forward(self, x):
        z = self.projection(x)
        p = self.projection(z)
        return z, p
