import torch
import torch.nn as nn
import torchvision.models as models
from typing import List
from resnet import resnet50
import encoder
from decoder import LightDecoder
from timm.layers import trunc_normal_


class SparkModel(nn.Module):
    def __init__(self, sparse_encoder: encoder.SparseEncoder, dense_decoder: LightDecoder, mask_ratio = 0.6):
        super(SparkModel, self).__init__()

        self.sparse_encoder = sparse_encoder
        self.dense_decoder = dense_decoder
        self.downsample_ratio = (16, 32, 32)
        self.input_size = sparse_encoder.input_size
        self.mask_ratio = mask_ratio
        self.fmap_d, self.fmap_h, self.fmap_w = self.input_size[0] // self.downsample_ratio[0], \
                                                self.input_size[1] // self.downsample_ratio[1], \
                                                self.input_size[2] // self.downsample_ratio[2] 
        
        self.len_keep = round(self.fmap_d * self.fmap_h * self.fmap_w * (1 - mask_ratio))
       
        '''
        self.num_feature_maps
            The number of feature maps created from encoder, ie, layers/deepness of the encoder

        self.densify_norms 
            List of normalization layers, one for each feature map

        self.densify_projs
            List of convolutional layers used to adjust feature map dimensions before passing to decoder 
        
        '''
        self.num_feature_maps = len(sparse_encoder.enc_feat_map_chs)
        self.densify_norms = nn.ModuleList()
        self.densify_projs = nn.ModuleList()
        self.mask_tokens = nn.ParameterList()

        # build the densify layers 
        encoder_channels_list : List[int] = sparse_encoder.enc_feat_map_chs
        decoder_channel : int = dense_decoder.channel_width 

        '''
        In the order from the smallest feature map 
        '''
        for i in range(self.num_feature_maps):
            channels = encoder_channels_list.pop()
            p = nn.Parameter(torch.zeros(1, decoder_channel, 1, 1, 1))
            trunc_normal_(p, mean=0, std=0.02, a=-0.02, b=0.02)
            self.mask_tokens.append(p)

            densify_norm = encoder.SparseBatchNorm3d(channels)
            self.densify_norms.append(densify_norm)

            # create densify proj
            if i == 0:
                assert channels == decoder_channel, f"the number of channels of the last encoder feature map {channels} needs to match the input channel size of decoder {decoder_channel}"
                densify_proj = nn.Identity()
            else: 
                kernel_size = 1 if i <= 0 else 3
                densify_proj = nn.Conv3d(
                    channels, decoder_channel, kernel_size = kernel_size, stride = 1, padding = kernel_size // 2, bias = True
                )

            self.densify_projs.append(densify_proj)

            decoder_channel //= 2
        
    def mask_3d(self, batch_size, device):
        '''
        Input image shape is torch.Size([batch, 1, 80, 256, 256])
        '''
        d, h, w = self.fmap_d, self.fmap_h, self.fmap_w
        idx = torch.rand(batch_size, d*h*w).argsort(dim=1)

        # B, len keep
        idx = idx[:, :self.len_keep].to(device)
        mask =  torch.zeros(batch_size, d*h*w, dtype=torch.bool, device=device).scatter_(dim=1, index = idx, value=True).view(batch_size, 1, d, h, w) 
        # print("mask dimension: ", mask.shape)
        return mask
    
    def forward(self, input, vis=False):
        '''
        STEP 0: MASKING THE INPUT

        mask: shape [batch, 1, d, h, w]
        '''
        mask = self.mask_3d(input.shape[0], input.device)
        encoder._mask = mask
        mask_enlarged = mask.repeat_interleave(self.downsample_ratio[0], 2).repeat_interleave(self.downsample_ratio[1], 3).repeat_interleave(self.downsample_ratio[2], 4)
        # print("enlarged mask dim", mask_enlarged.shape)
        # print("input dim", input.shape)

        masked = input * mask_enlarged
        
        ''' 
        STEP 1: ENCODING
        PASS MASKED IMAGE TO SPARSE ENCODER AND GET THE FEATURE MAPS 

        feature_map = [batch, channel, depth, height, width]
        '''
        feature_maps : List[torch.Tensor] = self.sparse_encoder(masked)
        feature_maps.reverse()

        # for map in feature_maps:
        #     print("shape of feature map: ", map.shape)  

        '''
        STEP 2: DENSIFICATION
        DENSITFY THE FEATURE MAPS WITH [MASK] TOKENS 

        dense_features: shape [batch, 1, d, h, w]
        '''

        cur_active = mask  
        densed = []

        for i, feature_map in enumerate(feature_maps):
            if feature_map is not None: 
                normalized_feature_map = self.densify_norms[i](feature_map)
                mask_tokens = self.mask_tokens[i].expand_as(normalized_feature_map)
                normalized_feature_map = torch.where(cur_active.expand_as(normalized_feature_map), normalized_feature_map, mask_tokens)
                normalized_feature_map = self.densify_projs[i](normalized_feature_map)
            densed.append(normalized_feature_map)
            cur_active = cur_active.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)

        '''
        STEP 3: DECODING
        USE A DENSE DECODER TO RECONSTRUCT THE INPUT
        '''
        # Decode and reconstruct using feature maps in reverse order
        reconstructed = self.dense_decoder(densed)

        inp_patches = self.patchify(input)
        rec_patches = self.patchify(reconstructed)
        mean = inp_patches.mean(dim=-1, keepdim=True)
        var = (inp_patches.var(dim=-1, keepdim=True) + 1e-6) ** 0.5
        inp_patches = (inp_patches - mean) / var

        # Compute L2 loss
        l2_loss = ((rec_patches - inp_patches) ** 2).mean(dim=-1)  # (B, L)

        # Loss on masked patches only
        mask_flat = mask.view(mask.shape[0], -1)  # (B, L)
        non_active = mask_flat.logical_not().float()
        recon_loss = (l2_loss * non_active).sum() / (non_active.sum() + 1e-8)


        if vis:
            # unpatchify reconstructed patches 
            reconstructed_patches = self.unpatchify(rec_patches * var + mean)
            '''
            Where mask_enlarged entry is true, it's unmasked, where false, it's masked. (0 values are the mask)
            '''
            hybrid = torch.where(mask_enlarged, input, reconstructed_patches)
            return input, masked, hybrid
        else:
            return recon_loss
    
    def patchify(self, bcdhw):
        """
        Divide 3D image into patches based on downsample ratio.
        """
        d_ratio, h_ratio, w_ratio = self.downsample_ratio
        d, h, w = self.fmap_d, self.fmap_h, self.fmap_w
        B, C = bcdhw.shape[:2]
        
        bcdhw = bcdhw.reshape(shape=(B, C, d, d_ratio, h, h_ratio, w, w_ratio))
        bcdhw = torch.einsum('bcdphqwr->bdhwpqrc', bcdhw)  # Rearrange axes
        
        bln = bcdhw.reshape(shape=(B, d * h *w, C * d_ratio * h_ratio * w_ratio))
        return bln
        
    def unpatchify(self, bln):
        d_ratio, h_ratio, w_ratio = self.downsample_ratio
        d, h, w = self.fmap_d, self.fmap_h, self.fmap_w

        B, C = bln.shape[0], bln.shape[-1] // (d_ratio * h_ratio * w_ratio)
        
        bln = bln.reshape(shape=(B, d, h, w, d_ratio, h_ratio, w_ratio, C))
        bln = torch.einsum('bdhwpqrc->bcdphqwr', bln)
        bchw = bln.reshape(shape=(B, C, d * d_ratio, h * h_ratio, w * w_ratio))
        return bchw

    def compute_loss(self, inp_patches, rec_patches, mask):
        """
        Compute reconstruction loss only on the masked regions.
        """

        # print("patchified shape", inp_patches.shape)
        # Normalize patches
        mean = inp_patches.mean(dim=-1, keepdim=True)
        var = (inp_patches.var(dim=-1, keepdim=True) + 1e-6) ** 0.5
        inp_patches = (inp_patches - mean) / var

        # Compute L2 loss
        l2_loss = ((rec_patches - inp_patches) ** 2).mean(dim=-1)  # (B, L)

        # Loss on masked patches only
        mask_flat = mask.view(mask.shape[0], -1)  # (B, L)
        non_active = mask_flat.logical_not().float()
        masked_loss = (l2_loss * non_active).sum() / (non_active.sum() + 1e-8)
        
        return masked_loss

