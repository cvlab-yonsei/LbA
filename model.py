import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from utils.model_utils import GeMP, resnet50


class CMAlign(nn.Module):
    def __init__(self, batch_size=8, num_pos=4, temperature=50):
        super(CMAlign, self).__init__()
        self.batch_size = batch_size
        self.num_pos = num_pos
        self.criterion = nn.TripletMarginLoss(margin=0.3, p=2.0, reduce=False)
        self.temperature = temperature

    def _random_pairs(self):
        batch_size = self.batch_size
        num_pos = self.num_pos

        pos = []
        for batch_index in range(batch_size):
            pos_idx = random.sample(list(range(num_pos)), num_pos)
            pos_idx = np.array(pos_idx) + num_pos*batch_index
            pos = np.concatenate((pos, pos_idx))
        pos = pos.astype(int)

        neg = []
        for batch_index in range(batch_size):
            batch_list = list(range(batch_size))
            batch_list.remove(batch_index)
            
            batch_idx = random.sample(batch_list, num_pos)
            neg_idx = random.sample(list(range(num_pos)), num_pos)

            batch_idx, neg_idx = np.array(batch_idx), np.array(neg_idx)
            neg_idx = batch_idx*num_pos + neg_idx
            neg = np.concatenate((neg, neg_idx))
        neg = neg.astype(int)

        return {'pos': pos, 'neg': neg}

    def _define_pairs(self):
        pairs_half1 = self._random_pairs()
        pos_half1, neg_half1 = pairs_half1['pos'], pairs_half1['neg']

        pairs_half2 = self._random_pairs()
        pos_half2, neg_half2 = pairs_half2['pos'], pairs_half2['neg']
        
        pos_half2 += self.batch_size*self.num_pos
        neg_half2 += self.batch_size*self.num_pos

        return {'pos': np.concatenate((pos_half2, pos_half1)), 'neg': np.concatenate((neg_half2, neg_half1))}

    def feature_similarity(self, feat_source, feat_target):
        batch_size, fdim, h, w = feat_source.shape
        assert feat_source.shape == feat_target.shape
        
        feat_target = feat_target.view(batch_size, fdim, -1)
        feat_source = feat_source.view(batch_size, fdim, -1)

        feat_target = F.normalize(feat_target, p=2.0, dim=1, eps=1e-12).permute(0,2,1)
        feat_source = F.normalize(feat_source, p=2.0, dim=1, eps=1e-12)

        feature_similarity = torch.bmm(feat_target, feat_source)
        return feature_similarity

    def matching_probability(self, feature_similarity):
        return F.softmax(feature_similarity*self.temperature, dim=2)

    def soft_warping(self, matching_probability, feat_source):
        batch_size, fdim, h, w = feat_source.shape
        feat_flatten = feat_source.view(batch_size, fdim, -1).permute(0,2,1)
        warped_flatten = torch.bmm(matching_probability, feat_flatten)
        return warped_flatten.permute(0,2,1).view(batch_size, fdim, h, w)

    def reconstruction(self, feat_warped, feat_target):
        mask = self.compute_mask(feat_target)
        reconstruction = mask*feat_warped + (1-mask)*feat_target
        return reconstruction

    def compute_mask(self, feat):
        batch_size, fdim, h, w = feat.shape
        feat = feat.view(batch_size, fdim, -1)
        norm = torch.norm(feat, dim=1)

        # min-max normalization
        norm -= norm.min(dim=1, keepdim=True)[0]
        norm /= norm.max(dim=1, keepdim=True)[0] + 1e-12

        mask = norm.view(batch_size, -1, h, w)
        return mask.detach()

    def compute_comask(self, matching_probability, mask_source, mask_target):
        batch_size, fdim, h, w = mask_target.shape
        assert mask_source.shape == mask_target.shape

        mask_source_flatten = mask_source.view(batch_size, fdim, -1).permute(0,2,1)
        mask_source_warp_flatten = torch.bmm(matching_probability, mask_source_flatten).detach()
        
        comask = mask_source_warp_flatten.permute(0,2,1).view(batch_size, fdim, h, w) * mask_target
        comask_flatten = comask.view(batch_size, -1)

        # min-max normalization
        comask_flatten -= comask_flatten.min(dim=1, keepdim=True)[0]
        comask_flatten /= comask_flatten.max(dim=1, keepdim=True)[0] + 1e-12

        comask = comask_flatten.view(batch_size, fdim, h, w)
        return comask

    def _forward(self, feat):
        pairs = self._define_pairs()
        pos_idx, neg_idx = pairs['pos'], pairs['neg']

        # positive
        feature_similarity = self.feature_similarity(feat, feat[pos_idx])
        matching_probability = self.matching_probability(feature_similarity)
        warped_feat = self.soft_warping(matching_probability, feat)
        recon_feat_pos = self.reconstruction(warped_feat, feat)

        mask_source = self.compute_mask(feat)
        comask = self.compute_comask(matching_probability, mask_source, mask_source[pos_idx])

        # negative
        feature_similarity = self.feature_similarity(feat, feat[neg_idx])
        matching_probability = self.matching_probability(feature_similarity)
        warped_feat = self.soft_warping(matching_probability, feat)
        recon_feat_neg = self.reconstruction(warped_feat, feat)

        # compute dense triplet loss
        loss = torch.mean(comask * self.criterion(feat, recon_feat_pos, recon_feat_neg))

        return recon_feat_pos, loss
        
    def forward(self, feat_v, feat_t):
        feat = torch.cat([feat_v, feat_t], dim=0)
        recon, loss = self._forward(feat)
        return {'feat': recon, 'loss': loss}

# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, args):
        super(base_resnet, self).__init__()
        
        self.args = args
        base = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.base = base
        
    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        feat4 = self.base.layer4(x)
        return {'feat3': x, 'feat4': feat4}


class embed_net(nn.Module):
    def __init__(self, args, class_num):
        super(embed_net, self).__init__()
        self.args = args
        self.thermal_module = thermal_module()
        self.visible_module = visible_module()
        self.base_resnet = base_resnet(self.args)

        pool_dim = 2048
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.pool = GeMP()

        self.cmalign = CMAlign(args.batch_size, args.num_pos)


    def forward(self, x_v, x_t, modal=0, use_cmalign=True):
        if modal == 0:
            x_v = self.visible_module(x_v)
            x_t = self.thermal_module(x_t)
            x = torch.cat((x_v, x_t), 0)
        elif modal == 1:
            x = self.visible_module(x_v)
        elif modal == 2:
            x = self.thermal_module(x_t)

        feat = self.base_resnet(x)

        if use_cmalign:
            ### layer3
            feat3 = feat['feat3']
            batch_size, fdim, h, w = feat3.shape
            out3 = self.cmalign(feat3[:batch_size//2], feat3[batch_size//2:])

            feat3_recon = self.base_resnet.base.layer4(out3['feat'])
            feat3_recon_p = self.pool(feat3_recon)
            cls_ic_layer3 = self.classifier(self.bottleneck(feat3_recon_p))

            ### layer4
            feat4 = feat['feat4']
            feat4_p = self.pool(feat4)
            batch_size, fdim, h, w = feat4.shape
            out4 = self.cmalign(feat4[:batch_size//2], feat4[batch_size//2:])

            feat4_recon = out4['feat']
            feat4_recon_p = self.pool(feat4_recon)
            cls_ic_layer4 = self.classifier(self.bottleneck(feat4_recon_p))

            ### compute losses
            cls_id = self.classifier(self.bottleneck(feat4_p))
            loss_dt = out3['loss'] + out4['loss']

            return {
                'feat4_p': feat4_p,
                'cls_id': cls_id, 
                'cls_ic_layer3': cls_ic_layer3, 
                'cls_ic_layer4': cls_ic_layer4, 
                'loss_dt': loss_dt
                }

        else:
            feat4 = feat['feat4']
            batch_size, fdim, h, w = feat4.shape
            feat4_flatten = feat['feat4'].view(batch_size, fdim, -1)
            feat4_p = self.pool(feat4_flatten)
            cls_id = self.classifier(self.bottleneck(feat4_p))
            return {
                'feat4_p': feat4_p,
                'cls_id': cls_id,
                'feat4_p_norm': F.normalize(feat4_p, p=2.0, dim=1)
            }
