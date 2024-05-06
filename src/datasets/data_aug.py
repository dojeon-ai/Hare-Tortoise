from torchvision.transforms import transforms

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


