from .backbones import *  
from .heads import * 
from .base import Model
from omegaconf import OmegaConf
from src.common.class_utils import all_subclasses
import torch


BACKBONES = {subclass.get_name():subclass
            for subclass in all_subclasses(BaseBackbone)}

HEADS = {subclass.get_name():subclass
         for subclass in all_subclasses(BaseHead)}


def build_model(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    backbone_cfg = cfg['backbone']
    head_cfg = cfg['head']
    
    backbone_type = backbone_cfg.pop('type')
    head_type = head_cfg.pop('type')

    # backbone
    backbone_cls = BACKBONES[backbone_type]
    backbone = backbone_cls(**backbone_cfg)
    fake_obs = torch.zeros((2, *backbone_cfg['in_shape']))
    out = backbone(fake_obs)

    # head
    head_cfg['in_shape'] = out.shape[1:]
    head_cls = HEADS[head_type]
    head = head_cls(**head_cfg)
    out = head(out)
    
    # model
    model = Model(backbone=backbone, head=head)
    
    return model
