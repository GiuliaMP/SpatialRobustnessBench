# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
# from torchvision.models import resnet50, resnet101, resnet152

from ptsemseg.models.utils_for_bisenet.xception import xception39
from ptsemseg.models.utils_for_bisenet.seg_oprs import ConvBnRelu, AttentionRefinement, FeatureFusion, AdaptiveAvgPool2dWithMask




class BISENET_X39(nn.Module):

    def __init__(self, num_classes=19, is_training=None,
                 criterion=None, ohem_criterion=None, pretrained_model=None,
                 norm_layer=nn.BatchNorm2d):
        super(BISENET_X39, self).__init__()
        self.context_path = xception39(pretrained_model, norm_layer=norm_layer)

        self.mask = None
        self.trigger = None

        self.business_layer = []
        self.is_training = is_training

        self.spatial_path = SpatialPath(3, 128, norm_layer)

        out_planes = num_classes
        conv_channel = 128

        
        self.use_attention = True
        self.global_context = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), 
                ConvBnRelu(256, conv_channel, 1, 1, 0,
                        has_bn=True,
                        has_relu=True, has_bias=False, norm_layer=norm_layer)  
            )

        print("attention: " + str(self.use_attention))

        # stage = [256, 128, 64]
        arms = [AttentionRefinement(256, conv_channel, norm_layer),
                AttentionRefinement(128, conv_channel, norm_layer)]
        refines = [ConvBnRelu(conv_channel, conv_channel, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, has_bias=False),
                   ConvBnRelu(conv_channel, conv_channel, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, has_bias=False)]
        heads = [BiSeNetHead(conv_channel, out_planes, 16,
                             True, norm_layer),
                 BiSeNetHead(conv_channel, out_planes, 8,
                             True, norm_layer),
                 BiSeNetHead(conv_channel * 2, out_planes, 8,
                             False, norm_layer)]
        self.ffm = FeatureFusion(conv_channel * 2, conv_channel * 2,
                                 1, norm_layer)

        self.arms = nn.ModuleList(arms)
        self.refines = nn.ModuleList(refines)
        self.heads = nn.ModuleList(heads)

        self.business_layer.append(self.spatial_path)
        self.business_layer.append(self.global_context)
        self.business_layer.append(self.arms)
        self.business_layer.append(self.refines)
        self.business_layer.append(self.heads)
        self.business_layer.append(self.ffm)

        if is_training:
            self.criterion = criterion
            self.ohem_criterion = ohem_criterion

    
    def set_attention(self,use_attention):
        self.use_attention = use_attention


    def forward(self, data, label=None):
        spatial_out = self.spatial_path(data)

        context_blocks = self.context_path(data)
        context_blocks.reverse()

        if self.use_attention is True:
            global_context = self.global_context(context_blocks[0])
            global_context = F.interpolate(global_context,
                                                    size=context_blocks[0].size()[2:],
                                                    mode='bilinear', align_corners=True)
            last_fm = global_context
        
        pred_out = []
        for i, (fm, arm, refine) in enumerate(zip(context_blocks[:2], self.arms,
                                                  self.refines)):
            #-------------------------------------
            # added by giulio
            fm = arm(fm, use_attention=self.use_attention)
            if self.use_attention is True or i > 0:
                fm += last_fm
            #-------------------------------------

            last_fm = F.interpolate(fm, size=(context_blocks[i + 1].size()[2:]),
                                    mode='bilinear', align_corners=True)
            last_fm = refine(last_fm)
            pred_out.append(last_fm)
        context_out = last_fm

        concate_fm = self.ffm(spatial_out, context_out)
        # concate_fm = self.heads[-1](concate_fm)
        pred_out.append(concate_fm)

        if self.is_training:
            aux_loss0 = self.ohem_criterion(self.heads[0](pred_out[0]), label)
            aux_loss1 = self.ohem_criterion(self.heads[1](pred_out[1]), label)
            main_loss = self.ohem_criterion(self.heads[-1](pred_out[2]), label)
            loss = + aux_loss0 + aux_loss1
            return loss

        return F.log_softmax(self.heads[-1](pred_out[-1]), dim=1)



    def set_mask(self, mask, layer):
        mask_ = self.mask
        mask_ = F.interpolate(mask_, size=layer.shape[2:],mode='bilinear', align_corners=True)
        mask_ = torch.heaviside(mask_ - 0.5, values=torch.tensor([0.0]).to('cuda'))
        layer = layer * (1 - mask_)
        #std, mean = torch.std_mean(layer, unbiased=False)
        #layer += (mask_ * (torch.randn(size=mask_.size()).to('cuda') * std + mean))
        return layer


    def protected_forward(self, data, defense_module = None, label=None):
        spatial_out = self.spatial_path(data)

        context_blocks = self.context_path(data)
        context_blocks.reverse()

        #-----------------------------------------------------------------------------
        # checkpoint 
        self.trigger, self.mask = defense_module(context_blocks[1])
        self.mask *= self.trigger#.view(self.trigger.shape[0], 1) 
        
        # cleaning part (everything that is linked with the checkpoint)
        
        context_blocks[1] = self.set_mask(self.mask, context_blocks[1])
        context_blocks[0] = self.context_path.layer3(context_blocks[1])

        spatial_out = self.set_mask(self.mask, spatial_out)
        
        #-----------------------------------------------------------------------------

        if self.use_attention is True:
            global_context = self.global_context(context_blocks[0])
            global_context = F.interpolate(global_context,
                                                    size=context_blocks[0].size()[2:],
                                                    mode='bilinear', align_corners=True)
            global_context = self.set_mask(self.mask, global_context)
            last_fm = global_context
        
        pred_out = []
        for i, (fm, arm, refine) in enumerate(zip(context_blocks[:2], self.arms,
                                                  self.refines)):
            #-------------------------------------
            # added by giulio
            fm = arm(context_blocks[i], use_attention=self.use_attention, mask=self.mask)
            if self.use_attention is True or i > 0:
                fm += last_fm
            #-------------------------------------

            last_fm = F.interpolate(fm, size=(context_blocks[i + 1].size()[2:]),
                                    mode='bilinear', align_corners=True)
            last_fm = self.set_mask(self.mask, last_fm)
            last_fm = refine(last_fm)
            pred_out.append(last_fm)

        context_out = last_fm
        context_out = self.set_mask(self.mask, context_out)
        concate_fm = self.ffm(spatial_out, context_out, mask=self.mask)
        # concate_fm = self.heads[-1](concate_fm)
        pred_out.append(concate_fm)

        if self.is_training:
            aux_loss0 = self.ohem_criterion(self.heads[0](pred_out[0]), label)
            aux_loss1 = self.ohem_criterion(self.heads[1](pred_out[1]), label)
            main_loss = self.ohem_criterion(self.heads[-1](pred_out[2]), label)
            loss = + aux_loss0 + aux_loss1
            return loss

        #------------------------------------------------------------------------
        # clean the output using the extracted mask
        #pred_out[-1] = self.set_mask(self.mask, pred_out[-1])
        out = self.heads[-1](pred_out[-1], up_layer=self)
        #out = self.set_mask(self.mask, out)
        return F.log_softmax(out, dim=1)
        #------------------------------------------------------------------------





    #----------------------------------------------------------------------------
    # Needed for training the detector
    #----------------------------------------------------------------------------
    def forward_features_context_blocks_0(self, data, label=None):
        spatial_out = self.spatial_path(data)
        context_blocks = self.context_path(data)
        context_blocks.reverse()
        return context_blocks[0]

        

    def forward_features_context_blocks_1(self, data, label=None):
        spatial_out = self.spatial_path(data)
        context_blocks = self.context_path(data)
        context_blocks.reverse()
        return context_blocks[1]



    def forward_features_pred_out_0(self, data, label=None,):
        spatial_out = self.spatial_path(data)

        context_blocks = self.context_path(data)
        context_blocks.reverse()

        if self.use_attention is True:
            global_context = self.global_context(context_blocks[0])
            global_context = F.interpolate(global_context,
                                                    size=context_blocks[0].size()[2:],
                                                    mode='bilinear', align_corners=True)
            last_fm = global_context
        
        pred_out = []
        for i, (fm, arm, refine) in enumerate(zip(context_blocks[:2], self.arms,
                                                  self.refines)):
            #-------------------------------------
            # added by giulio
            fm = arm(fm, use_attention=self.use_attention)
            if self.use_attention is True or i > 0:
                fm += last_fm
            #-------------------------------------

            last_fm = F.interpolate(fm, size=(context_blocks[i + 1].size()[2:]),
                                    mode='bilinear', align_corners=True)
            last_fm = refine(last_fm)
            pred_out.append(last_fm)
        return pred_out[0]



    def forward_features_pred_out_1(self, data, label=None,):
        spatial_out = self.spatial_path(data)

        context_blocks = self.context_path(data)
        context_blocks.reverse()

        if self.use_attention is True:
            global_context = self.global_context(context_blocks[0])
            global_context = F.interpolate(global_context,
                                                    size=context_blocks[0].size()[2:],
                                                    mode='bilinear', align_corners=True)
            last_fm = global_context
        
        pred_out = []
        for i, (fm, arm, refine) in enumerate(zip(context_blocks[:2], self.arms,
                                                  self.refines)):
            #-------------------------------------
            # added by giulio
            fm = arm(fm, use_attention=self.use_attention)
            if self.use_attention is True or i > 0:
                fm += last_fm
            #-------------------------------------

            last_fm = F.interpolate(fm, size=(context_blocks[i + 1].size()[2:]),
                                    mode='bilinear', align_corners=True)
            last_fm = refine(last_fm)
            pred_out.append(last_fm)
        return pred_out[1]







class SpatialPath(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d):
        super(SpatialPath, self).__init__()
        inner_channel = 64
        self.conv_7x7 = ConvBnRelu(in_planes, inner_channel, 7, 2, 3,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_3x3_1 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                     has_bn=True, norm_layer=norm_layer,
                                     has_relu=True, has_bias=False)
        self.conv_3x3_2 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                     has_bn=True, norm_layer=norm_layer,
                                     has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(inner_channel, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        output = self.conv_1x1(x)

        return output


class BiSeNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BiSeNetHead, self).__init__()
        if is_aux:
            self.conv_3x3 = ConvBnRelu(in_planes, 128, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        else:
            self.conv_3x3 = ConvBnRelu(in_planes, 64, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        # self.dropout = nn.Dropout(0.1)
        if is_aux:
            self.conv_1x1 = nn.Conv2d(128, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(64, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        self.scale = scale

    def forward(self, x, up_layer = None):
        fm = self.conv_3x3(x)
        # fm = self.dropout(fm)
        output = self.conv_1x1(fm)
        if self.scale > 1:
            output = F.interpolate(output, scale_factor=self.scale,
                                   mode='bilinear',
                                   align_corners=True)

        return output


if __name__ == "__main__":
    model = BiSeNet(19, None)
    # print(model)