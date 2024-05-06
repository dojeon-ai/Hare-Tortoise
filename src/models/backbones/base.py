from abc import *
import torch.nn as nn
import torch

class BaseBackbone(nn.Module, metaclass=ABCMeta):
    def __init__(self, in_shape):
        super().__init__()
        self.in_shape = in_shape

    @classmethod
    def get_name(cls):
        return cls.name

    @abstractmethod
    def forward(self, x):
        """
        [param] x (torch.Tensor): (n, c, h, w)
        [return] x (torch.Tensor): (n, c, h, w)
        """
        pass
    
    @property
    def output_dim(self):
        pass