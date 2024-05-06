from .base import BaseHead
from .gap import GAPHead
from .spatial_pool import SpatialPoolHead
from .mlp import MLPHead

__all__ = [
    'BaseHead', 
    'SpatialPoolHead', 
    'GAPHead',
    'MLPHead'
]