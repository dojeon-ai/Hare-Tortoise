from abc import *
import torch
import torch.nn as nn
from .base import BaseHead
from einops import rearrange
from src.models.layers import init_normalization, init_activation


class SpatialPoolHead(BaseHead):
    name = 'spatial_pool'
    def __init__(self, 
                 in_shape, 
                 output_size,
                 pool_size,
                 norm_type,
                 activ_type,
                 drop_prob,
                 hidden_dims):
        
        super().__init__(in_shape, output_size)
        c, h, w = self.in_shape
        self.spatial_embed = nn.Embedding(1, c*h*w)
        self.pool = nn.AvgPool2d(pool_size)
        self.norm = init_normalization(channels=c, norm_type=norm_type)
        self.activ = init_activation(activ_type=activ_type)
        in_channel = c * (h//pool_size) * (w//pool_size)
        
        # todo in channel declare
        in_channel = c
        layers = []
        hidden_dims = [in_channel] + hidden_dims
        for idx in range(len(hidden_dims)-1):
            h_in = hidden_dims[idx]
            h_out = hidden_dims[idx+1]
            if activ_type == 'crelu':
                h_out = h_out // 2
            layers.append(nn.Linear(h_in, h_out))
            layers.append(self.activ)
            layers.append(nn.Dropout(p=drop_prob))
        
        layers.append(nn.Linear(hidden_dims[-1], output_size))
        self.fc = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        [params] x (torch.Tensor: (n, c, h, w))
        [returns] x (torch.Tensor: (n, d))
        """
        n, c, h, w = x.shape
        idx = torch.zeros(n, device=x.device).long()
        spatial_embed = self.spatial_embed(idx)
        spatial_embed = rearrange(spatial_embed, 'n (c h w) -> n c h w', c=c, h=h, w=w)
        x = x * spatial_embed
        x = self.pool(x)
        
        x = rearrange(x, 'n c 1 1 -> n c')
        x = self.norm(x)
        x = self.fc(x)        
                
        return x