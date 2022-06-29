import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
from aggregation_zeropad import LocalConvolution
##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images, paddings


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    paddings = (0, 0, 0, 0)

    if padding == 'same':
        images, paddings = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches, paddings

def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


class ContextualAttention_Enhance(nn.Module):
    def __init__(self, ksize=7, stride_1=4, stride_2=1, softmax_scale=10,shape=64 ,p_len=64,in_channels=64
                 , inter_channels=64,use_multiple_size=False,add_soft=0.01,use_topk=False,add_SE=False):
        super(ContextualAttention_Enhance, self).__init__()
        self.ksize = ksize
        self.shape=shape
        self.p_len=p_len
        self.add_soft=add_soft
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.softmax_scale = softmax_scale        
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.use_multiple_size=use_multiple_size
        self.relu=nn.LeakyReLU(negative_slope=0.1,inplace=True)
        self.use_topk=use_topk
        self.add_SE=add_SE
        # self.SE=SE_net(in_channels=in_channels)
        # self.change_channel=conv(64,128,kernel_size=3, stride=2)
        self.conv33=nn.Conv2d(in_channels=2*in_channels,out_channels=in_channels,kernel_size=1,stride=1,padding=0)
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                            padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)

    def forward(self, b):
        kernel = self.ksize
       
        b1 = self.relu(self.g(b))
        b2 =self.relu(self.theta(b))
        b3 =self.relu(self.phi(b))

        raw_int_bs = list(b1.size())  # b*c*h*w

        patch_28, paddings_28 = extract_image_patches(b1, ksizes=[self.ksize, self.ksize],strides=[self.stride_1, self.stride_1],rates=[1, 1],padding='same')
        patch_28 = patch_28.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        patch_28 = patch_28.permute(0, 4, 1, 2, 3)
        patch_28_group = torch.split(patch_28, 1, dim=0)

        patch_112, paddings_112 = extract_image_patches(b2, ksizes=[self.ksize, self.ksize],strides=[self.stride_2, self.stride_2],rates=[1, 1],padding='same')
        patch_112 = patch_112.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        patch_112 = patch_112.permute(0, 4, 1, 2, 3)
        patch_112_group = torch.split(patch_112, 1, dim=0)

        patch_112_2, paddings_112_2 = extract_image_patches(b3, ksizes=[self.ksize, self.ksize],strides=[self.stride_2, self.stride_2],rates=[1, 1],padding='same')
        patch_112_2 = patch_112_2.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        patch_112_2 = patch_112_2.permute(0, 4, 1, 2, 3)
        patch_112_group_2 = torch.split(patch_112_2, 1, dim=0)
        
        f_groups = torch.split(b3, 1, dim=0)
        y = []
        for xii,xi, wi,pi in zip(f_groups,patch_112_group_2, patch_28_group, patch_112_group):
            w,h = xii.shape[2], xii.shape[3]
            _, paddings = same_padding(xii, [self.ksize, self.ksize], [1, 1], [1, 1])
            wi_norm = wi[0]  # [L, C, k, k] 
            max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),axis=[1, 2, 3],keepdim=True)),self.escape_NaN)
            wi=wi/max_wi
            c_s = pi.shape[2]
            k_s = wi[0].shape[2]
            wi = wi.view(wi.shape[0],wi.shape[1],-1)
            xi = xi.permute(0, 2, 3, 4, 1)
            xi = xi.view(xi.shape[0],-1,xi.shape[4])
            score_map = torch.matmul(wi,xi)
            score_map = score_map.view(score_map.shape[0],score_map.shape[1],w,h)
            b_s, l_s, h_s, w_s = score_map.shape

            yi = score_map.view(b_s, l_s, -1)
            yi = F.softmax(yi*self.softmax_scale, dim=2).view(l_s, -1)
            pi = pi.view(h_s * w_s, -1)
            yi = torch.matmul(yi, pi)
            yi=yi.view(b_s, l_s, c_s, k_s, k_s)[0]
            zi = yi.view(1, l_s, -1).permute(0, 2, 1)
            zi = torch.nn.functional.fold(zi, (raw_int_bs[2], raw_int_bs[3]), (kernel, kernel), padding=paddings[0], stride=self.stride_1)
            
            inp = torch.ones_like(zi)
            inp_unf = torch.nn.functional.unfold(inp, (kernel, kernel), padding=paddings[0], stride=self.stride_1)
            out_mask = torch.nn.functional.fold(inp_unf, (raw_int_bs[2], raw_int_bs[3]), (kernel, kernel), padding=paddings[0], stride=self.stride_1)
            zi = zi / out_mask            
            y.append(zi)

        y = torch.cat(y, dim=0)
        y = self.W(y)
        y = b + y
        if self.add_SE:
            y_SE=self.SE(y)
            y=self.conv33(torch.cat((y_SE*y,y),dim=1))
        return y


class ContextualAttention_CotLayer_attention(nn.Module):
    def __init__(self, ksize=3, stride_1=1, stride_2=1, softmax_scale=10,in_channels=64, inter_channels=64):
        super(ContextualAttention_CotLayer_attention, self).__init__()
        self.ksize = ksize
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.softmax_scale = softmax_scale        
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        share_planes = 8
        factor = 2

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,padding=0)


        self.key_embed = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, self.ksize, stride=1, padding=self.ksize//2, groups=4, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)            
        )

        self.embed = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels//factor, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels//factor, pow(self.ksize, 2) * in_channels// share_planes, kernel_size=1),
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
        )

        self.local_conv = LocalConvolution(in_channels, in_channels, kernel_size=self.ksize, stride=1, padding=(self.ksize - 1) // 2, dilation=1)
        self.local_conv2 = LocalConvolution(in_channels, in_channels, kernel_size=self.ksize, stride=1, padding=(self.ksize - 1) // 2, dilation=1)
        # self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # reduction_factor = 4
        # self.radix = 2
        # attn_chs = max(in_channels * self.radix // reduction_factor, 32)
        # self.se = nn.Sequential(
        #     nn.Conv2d(in_channels, attn_chs, 1),
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #     nn.Conv2d(attn_chs, self.radix*in_channels, 1))
        
        self.second=nn.Conv2d(in_channels*2,in_channels,1)
        self.second_wei=nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels//factor, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels//factor, pow(self.ksize, 2) * in_channels// share_planes, kernel_size=1),)
        
    def forward(self, x):        
        original=x.clone()
        key=self.phi(x)
        query=self.theta(x)

        
        key = self.key_embed(key)
        qk = torch.cat([query, key], dim=1)
        b, c, qk_hh, qk_ww = qk.size()

        w = self.embed(qk)
        w = w.view(b, 1, -1, self.ksize*self.ksize, qk_hh, qk_ww)
        
        x = self.conv1x1(x)
        x = self.local_conv(x, w)



        x = torch.cat([x, key], dim=1)
        x_attn=self.second_wei(x)
        x_attn = F.softmax(x_attn, dim=2)
        w2= x_attn.view(b, 1, -1, self.ksize*self.ksize, qk_hh, qk_ww)
        out=self.local_conv2(self.second(x),w2) 
        #sec_       
        # B, C, H, W = x.shape
        # x = x.view(B, C, 1, H, W)
        # key = key.view(B, C, 1, H, W)
        # x = torch.cat([x, key], dim=2)
        # x_gap = x.sum(dim=2)
        # x_gap = x_gap.mean((2, 3), keepdim=True)
        # x_attn = self.se(x_gap)
        # x_attn = x_attn.view(B, C, self.radix)
        # x_attn = F.softmax(x_attn, dim=2)
        # out = (x * x_attn.reshape((B, C, self.radix, 1, 1))).sum(dim=2)        
        out=out +original
        return out.contiguous()


