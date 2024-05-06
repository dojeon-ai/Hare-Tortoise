from abc import *
import torch.nn as nn
import torch


class BaseHead(nn.Module, metaclass=ABCMeta):
    def __init__(self, in_shape, output_size):
        super().__init__()
        self.in_shape = in_shape
        self.output_size = output_size

    @classmethod
    def get_name(cls):
        return cls.name
