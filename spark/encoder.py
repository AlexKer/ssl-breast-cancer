import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import torch
import torch.nn as nn


'''
Dimension of the mask is [batch, 1, num_patch_d, num_patch_h, num_patch_w]
    ex: [batch, 1, 16, 32, 32]
'''
_mask: torch.Tensor = None  

'''
Function that upsamples the mask to match the size of the feature map so that masked convolution can occur. 
    D, H, W are the dimensions for the current feature map that the mask is being applied to. 
    returning_active_ex - boolean indicating whether or not to return the upsampled mask. if false, returns the indices of the non-mask regions instead


    active_ex should be shape: torch.Size([batch, 80, 128, 128]), ([batch, 40, 64, 64]),  ([batch, 20, 32, 32]),  ([batch, 10, 16, 16]), ([batch, 5, 8, 8]) 

    non_mask_indices: tuple of 4. 
        The first tensor in the tuple contains the batch indices of the non-zero elements.
        The second tensor contains the depth indices (corresponding to the 80 slices).
        The third tensor contains the height indices (from the 250 height dimension).
        The fourth tensor contains the width indices (from the 250 width dimension).
'''
def get_active_ex_or_ii(D, H, W, returning_active_ex=True):
    d_repeat, h_repeat, w_repeat = D // _mask.shape[-3], H // _mask.shape[-2], W // _mask.shape[-1]
    
    # print("d , h, w repeat", d_repeat, h_repeat, w_repeat)
    # print("shape of mask", _mask.shape)
    active_ex = _mask.repeat_interleave(d_repeat, dim=2).repeat_interleave(h_repeat, dim=3).repeat_interleave(w_repeat, dim=4)
    if returning_active_ex:
        active_ex = active_ex
        # print("shape of upsampled mask: ", active_ex.shape)
        return active_ex
    else:
        non_mask_indices = active_ex.squeeze(1).nonzero(as_tuple = True)
        # print("shape of non mask indices", non_mask_indices)
        return  non_mask_indices


'''
    Perform standard convolution. 
    Get upsampled mask. 
    Mask feature map.

    After first forward:  torch.Size([1, 64, 80, 128, 128])
'''
def sp_conv_forward(self, x: torch.Tensor):
    x = super(type(self), self).forward(x)
    
    active_mask = get_active_ex_or_ii(D= x.shape[2], H = x.shape[3], W = x.shape[4], returning_active_ex=True)
    # print("shape of x", x.shape)
    x *= active_mask  
    return x

'''
    indices is a tuple of length 4 where the first tensor contains the batch-dim indices, the second the depth, the third the height, the fourth the width, indices 
    for the non-mask entries

    x : shape [batch, channel, depth, h, w]
    permuted_x : [batch, d, h, w, c]
    non_mask: [num_non_mask, c]
'''
def sp_bn_forward(self, x: torch.Tensor):
    indices = get_active_ex_or_ii(D= x.shape[2], H = x.shape[3], W = x.shape[4], returning_active_ex=False)
    permuted_x = x.permute(0, 2, 3, 4, 1)
    
    non_mask = permuted_x[indices] 
    
    normalized_non_mask = super(type(self), self).forward(non_mask) 
    
    permuted_x = torch.zeros_like(permuted_x)
    permuted_x[indices] = normalized_non_mask
    original_perm = permuted_x.permute(0, 4, 1, 2, 3)
    return original_perm


class SparseConv3d(nn.Conv3d):
    forward = sp_conv_forward

class SparseMaxPooling(nn.MaxPool3d):
    forward = sp_conv_forward

class SparseAvgPooling(nn.AvgPool3d):
    forward = sp_conv_forward


class SparseConvNeXtLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_second", sparse=True):
        super().__init__(normalized_shape, eps, elementwise_affine=True)
        self.data_format = data_format
        self.sparse = sparse

    def forward(self, x):
        if self.data_format == "channels_last":  # For (B, D, H, W, C) format
            if self.sparse:
                ii = get_active_ex_or_ii(D= x.shape[1], H = x.shape[2], W = x.shape[3], returning_active_ex=False)
                nc = x[ii]
                nc = super(SparseConvNeXtLayerNorm, self).forward(nc)
                output = torch.zeros_like(x)
                output[ii] = nc
                return output
            else:
                return super(SparseConvNeXtLayerNorm, self).forward(x)
        else:  # For (B, C, D, H, W) format
            if self.sparse:
                ii = get_active_ex_or_ii(D= x.shape[1], H = x.shape[2], W = x.shape[3],returning_active_ex=False)
                permuted_x = x.permute(0, 2, 3, 4, 1)
                non_mask = permuted_x[ii]
                non_mask = super(SparseConvNeXtLayerNorm, self).forward(non_mask)

                x = torch.zeros_like(permuted_x)
                x[ii] = non_mask
                return x.permute(0, 4, 1, 2, 3)
            else:
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                return self.weight[:, None, None] * x + self.bias[:, None, None]
            

class SparseBatchNorm3d(nn.BatchNorm1d):
    forward = sp_bn_forward 

class SparseEncoder(nn.Module):
    def __init__(self, cnn, input_size):
        super(SparseEncoder, self).__init__()
        self.sp_cnn = SparseEncoder.dense_model_to_sparse(m=cnn)
        self.input_size : tuple = input_size
        self.downsample_ratio : tuple = cnn.get_downsample_ratio
        self.enc_feat_map_chs : List[int] = [256, 512, 1024, 2048]

    @staticmethod
    def dense_model_to_sparse(m: nn.Module):
        oup = m
        if isinstance(m, nn.Conv3d):
            bias = m.bias is not None
            oup = SparseConv3d(
                m.in_channels, 
                m.out_channels,
                kernel_size=m.kernel_size, stride=m.stride, padding=m.padding,
                dilation=m.dilation, groups=m.groups, bias=bias, padding_mode=m.padding_mode,
            )
            oup.weight.data.copy_(m.weight.data)
            if bias:
                oup.bias.data.copy_(m.bias.data)
        elif isinstance(m, nn.MaxPool3d):
            oup = SparseMaxPooling(m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation, return_indices=m.return_indices, ceil_mode=m.ceil_mode)
        elif isinstance(m, nn.AvgPool3d):
            oup = SparseAvgPooling(m.kernel_size, m.stride, m.padding, ceil_mode=m.ceil_mode, count_include_pad=m.count_include_pad, divisor_override=m.divisor_override)
        elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm)):
            oup = SparseBatchNorm3d(m.weight.shape[0], eps=m.eps, momentum=m.momentum, affine=m.affine, track_running_stats=m.track_running_stats)
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
            oup.running_mean.data.copy_(m.running_mean.data)
            oup.running_var.data.copy_(m.running_var.data)
            oup.num_batches_tracked.data.copy_(m.num_batches_tracked.data)
        elif isinstance(m, nn.LayerNorm) and not isinstance(m, SparseConvNeXtLayerNorm):
            oup = SparseConvNeXtLayerNorm(m.weight.shape[0], eps=m.eps)
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
        
        for name, child in m.named_children():
            oup.add_module(name, SparseEncoder.dense_model_to_sparse(child))
        del m
        return oup

    '''
    Forward pass.
    Arg: 
        x: Masked 3d image in the shape [batch, 1, 80, 256, 256]
    Return: 
        list of feature maps, which get produced from resnet by passing hierarchical as true. 
    '''    
    def forward(self, x) -> List[torch.Tensor]:
        return self.sp_cnn(x, hierarchical=True)
