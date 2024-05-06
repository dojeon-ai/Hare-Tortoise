import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from src.models.backbones.base import BaseBackbone
from src.models.layers import init_normalization

class MLPBackbone(BaseBackbone):
    name='mlp'
    def __init__(self, 
                 in_shape,
                 norm_type,
                 hidden_dims):

        super(MLPBackbone, self).__init__(in_shape)
        c, h, w = in_shape
        self.norm_type = norm_type
        in_channel = c * h * w
        layers = []
        hidden_dims = [in_channel] + hidden_dims
        for idx in range(len(hidden_dims)-1):
            h_in = hidden_dims[idx]
            h_out = hidden_dims[idx+1]
            layers.append(nn.Linear(h_in, h_out))
            layers.append(init_normalization(h_out, one_d=True))
            layers.append(nn.ReLU(inplace=True))
        
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        n, c, h, w = x.shape
        x = rearrange(x, 'n c h w -> n (c h w)')
        x = self.fc(x)
        x = x.view(n, -1, 1, 1)  # Reshaping the tensor
        
        return x