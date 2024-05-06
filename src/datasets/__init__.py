import copy
import random
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from omegaconf import OmegaConf
from src.datasets.dataset import PartialImageFolder


def build_dataloader(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    
    # initialize dataset
    dataset_type = cfg['type']
    
    if dataset_type in ['tig', 'cifar10', 'cifar100', 'mnist', 'shvn']:
        train_cfg = cfg['train']
        test_cfg = cfg['test']
        
        #######################
        # transformations
        def compose_transform(cfg):    
            transform_list = []
            
            # crop
            crop_type = cfg['crop_type']
            if crop_type == 'random_resize':
                transform_list.append(transforms.RandomResizedCrop(cfg['crop_size']))
            elif crop_type == 'random_pad':
                transform_list.append(transforms.RandomCrop(cfg['crop_size'], padding=cfg['pad_size']))
            elif crop_type == 'center':
                if cfg.scale:
                    transforms_list.append(transforms.Resize(cfg['scale']))
                transforms_list.append(transforms.CenterCrop(cfg['crop_size']))
            elif crop_type == 'none':
                transform_list.append(transforms.Lambda(lambda x: x))
            else:
                raise ValueError(f"Invalid crop type: {crop_type}")
                
            # horizontal flip
            if cfg['horizontal_flip']:
                transform_list.append(transforms.RandomHorizontalFlip())
                
            # gray scale
            if cfg['gray_scale']:
                transform_list.append(transforms.Grayscale())
                
            # normalization
            transform_list.append(transforms.ToTensor())
            transform_list.append(transforms.Normalize(**cfg['normalize']))

            return transforms.Compose(transform_list)

        train_transforms = compose_transform(train_cfg.pop('transforms'))
        test_transforms = compose_transform(test_cfg.pop('transforms'))
        
        ########################
        # dataset
        # initialization
        num_tasks = cfg['num_tasks']    
        input_data_ratios = cfg['input_data_ratios']
        label_noise_ratios = cfg['label_noise_ratios']
        buffer_size = cfg['buffer_size']
        
        if len(input_data_ratios) != num_tasks:
            raise ValueError("number of tasks does not match the number of dataset")
        
        train_datasets = []
        prev_indices = []
        for task_idx in range(num_tasks):
            input_data_ratio = input_data_ratios[task_idx]
            label_noise_ratio = label_noise_ratios[task_idx]
            
            train_dataset = PartialImageFolder(
                **train_cfg,
                transform = train_transforms,
                input_data_ratio=input_data_ratio,
                label_noise_ratio=label_noise_ratio,
                prev_indices=prev_indices,
                buffer_size=buffer_size
            )
            train_datasets.append(train_dataset)
            
            # get previous datasets' indices to stack the datasets
            cur_indices = train_dataset.get_indices() 
            prev_indices = list(set(prev_indices + cur_indices))
            
        test_dataset = PartialImageFolder(
            **test_cfg,
            transform = test_transforms,
            input_data_ratio=1.0,
            label_noise_ratio=0.0,
            prev_indices=[],
            buffer_size=0
        )
        
        ########################
        # dataloader
        dataloader_cfg = cfg['dataloader']
        
        train_loaders = []
        for train_dataset in train_datasets:
            train_loader = DataLoader(
                train_dataset, 
                **dataloader_cfg,
                shuffle=True
            )
            train_loaders.append(train_loader)
        
        test_loader = DataLoader(
            test_dataset, 
            **dataloader_cfg,
            shuffle=False
        )

    else:
        raise NotImplemented
    
    return train_loaders, test_loader
