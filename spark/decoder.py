# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List

import torch
import torch.nn as nn
from timm.layers import trunc_normal_


class UNetBlock(nn.Module):
    def __init__(self, cin, cout, bn3d):
        """
        a UNet block with 2x up sampling
        """
        super().__init__()
        self.up_sample = nn.ConvTranspose3d(cin, cin, kernel_size=4, stride=2, padding=1, bias=True)

        self.conv = nn.Sequential(
            nn.Conv3d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False),
            bn3d(cin), 
            nn.ReLU6(inplace=True),
            nn.Conv3d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False), 
            bn3d(cout),
        )
    
    def forward(self, x):
        x = self.up_sample(x)
        # print("after upsample: ", x.shape)
        x = self.conv(x)
        
        return x


class LightDecoder(nn.Module):
    def __init__(self, up_sample_ratio = (16, 32, 32), channel_width=2048):
        super().__init__()
        self.channel_width = channel_width
        '''
        UPDATE TO A TUPLE (16, 32, 32)
        '''
        d_n , h_n, w_n = up_sample_ratio
      
        bn3d = nn.BatchNorm3d
        channels = [2048, 1024, 512, 256, 64]
      
        self.dec = nn.ModuleList([UNetBlock(cin, cout, bn3d) for (cin, cout) in zip(channels[:-1], channels[1:])])
        # print("len of self dec", len(self.dec))
        # for (cin, cout) in zip(channels[:-1], channels[1:]):
        #     print(cin, cout)
        self.proj = nn.Sequential(
            nn.ConvTranspose3d(channels[-1], channels[-1], kernel_size=(1, 4, 4), stride= (1, 2, 2), padding = (0, 1, 1), bias=True),
            nn.Conv3d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1, bias=False),
            bn3d(channels[-1]), 
            nn.ReLU6(inplace=True),
            nn.Conv3d(channels[-1], 1, kernel_size=3, stride=1, padding=1, bias=False), 
            bn3d(1),
        )
        
        
        self.initialize()
    '''
    after conv1:  torch.Size([12, 64, 80, 128, 128])
    after layer 1 torch.Size([12, 256, 40, 64, 64])
    after layer 2 torch.Size([12, 512, 20, 32, 32])
    after layer 3 torch.Size([12, 1024, 10, 16, 16])
    after layer 4 torch.Size([12, 2048, 5, 8, 8])
    '''
    def forward(self, to_dec: List[torch.Tensor]):
        
        x = 0
        for i, d in enumerate(self.dec):
            
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = self.dec[i](x)
            # print("decoded shape", x.shape)
        
        return self.proj(x)
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
