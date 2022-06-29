import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from common import (PixelShufflePack, ResidualBlockNoBN,flow_warp, make_layer)
from vsr_deblur_module import ContextualAttention_Enhance,ContextualAttention_CotLayer_attention
from common.submodules import *
from utils import logger
from registry import BACKBONES
import logging
from mmcv.utils import get_logger

def get_root_logger(log_file=None, log_level=logging.INFO):
    # root logger name: mmedit
    logger = get_logger(__name__.split('.')[0], log_file, log_level)
    return logger

@BACKBONES.register_module()
class BasicVSRNet_deblur(nn.Module):

    def __init__(self, mid_channels=64, num_blocks=10, spynet_pretrained=None):

        super().__init__()

        self.mid_channels = mid_channels

        self.conv1_1_cot = conv(self.mid_channels,128, kernel_size=3, stride=2)
    
        self.conv1_2_cot = resnet_block(128, kernel_size=3)
        self.conv1_3_cot = resnet_block(128, kernel_size=3)
        self.spynet = SPyNet(pretrained=spynet_pretrained)



        self.backward_resblocks = ResidualBlocksWithInputConv(3, mid_channels, num_blocks)
        self.flow_conv=nn.Conv2d(mid_channels,3,1,1,0,bias=True)

        self.backward_resblocks_wrap_cot=ResidualBlocksWithInputConv(mid_channels+64,mid_channels, 15)
        self.forward_resblocks_wrap_cot=ResidualBlocksWithInputConv(mid_channels+64,mid_channels,15)

        self.fusion_cot = nn.Conv2d(256,128, 1, 1, 0, bias=True)
        self.lrelu_cot = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.deblur_resblocks_cot=ResidualBlocksWithInputConv(128,128,20)
    

        self.upconv1_u = upconv(128,64)
        self.upconv1_2 = resnet_block(64, kernel_size=3)
        self.upconv1_1 = resnet_block(64, kernel_size=3)
        self.img_prd = conv(64, 3, kernel_size=3)


        self.IPT_forward_cot=ContextualAttention_CotLayer_attention(ksize=3,in_channels=64, inter_channels=64)
        self.IPT_backward_cot=ContextualAttention_CotLayer_attention(ksize=3,in_channels=64, inter_channels=64)
        self.final_IPT_cot=ContextualAttention_CotLayer_attention(ksize=5,in_channels=128, inter_channels=128)


    def check_if_mirror_extended(self, lrs):

        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs):


        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended: 
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs):
        n, t, c, h, w = lrs.size()
        assert h >= 64 and w >= 64, ('The height and width of inputs should be at least 64, 'f'but got {h} and {w}.')
        


        lrs_pre_fea = lrs.new_zeros(n,t,self.mid_channels, h, w)

        for i in range(0,t):
            lrs_pre_fea[:, i, :, :, :]=self.backward_resblocks(lrs[:, i, :, :, :])
            
        lrs_flow_conv= lrs.new_zeros(n,t,3, h, w)

        for i in range(0,t):
            lrs_flow_conv[:, i, :, :, :]=self.flow_conv(lrs_pre_fea[:, i, :, :, :])


        self.check_if_mirror_extended(lrs)


        flows_forward, flows_backward = self.compute_flow(lrs_flow_conv)
        

        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            feat_prop_pre=lrs_pre_fea[:, i, :, :, :]
            if i < t - 1: 
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
              

            feat_prop = torch.cat([feat_prop_pre, feat_prop], dim=1)
            feat_prop=self.backward_resblocks_wrap_cot(feat_prop)
            feat_prop= self.IPT_backward_cot(self.IPT_backward_cot(feat_prop))+feat_prop
           


            conv2_d_back = (self.conv1_1_cot(feat_prop))
            conv2_d_back = self.conv1_3_cot(self.conv1_2_cot(conv2_d_back))
            feat_prop_backward_final=conv2_d_back
            outputs.append(feat_prop_backward_final)

        outputs = outputs[::-1]


        feat_prop =torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs_flow_conv[:, i, :, :, :]
            feat_prop_pre_forward=lrs_pre_fea[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))


            feat_prop = torch.cat([feat_prop_pre_forward, feat_prop], dim=1)
            feat_prop=self.forward_resblocks_wrap_cot(feat_prop)
            feat_prop=self.IPT_forward_cot(self.IPT_forward_cot(feat_prop))+feat_prop


            conv2_d_forward = (self.conv1_1_cot(feat_prop))
            conv2_d_forward = self.conv1_3_cot(self.conv1_2_cot(conv2_d_forward))
            feat_prop_forward_final=conv2_d_forward

            out = torch.cat([outputs[i],feat_prop_forward_final], dim=1)
            out = self.lrelu_cot(self.fusion_cot(out))
            out=self.deblur_resblocks_cot(out)

            out=self.final_IPT_cot(self.final_IPT_cot(out))+out
            

            upconv1 = self.upconv1_1(self.upconv1_2(self.upconv1_u(out)))
            output_img = self.img_prd(upconv1)+ lr_curr 
            outputs[i] = output_img

        return torch.stack(outputs, dim=1)

    def init_weights(self, pretrained=None, strict=True):

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


class ResidualBlocksWithInputConv(nn.Module):

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()
        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels))
        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)


class SPyNet(nn.Module):
    """SPyNet network structure.
    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, pretrained):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.
        Note that in this function, the images are already resized to a
        multiple of 32.
        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).
        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    flow_warp(
                        supp[level],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.
        This function computes the optical flow from ref to supp.
        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).
        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            ConvModule(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=None))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].
        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)