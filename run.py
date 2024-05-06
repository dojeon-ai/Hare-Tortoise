import argparse
from dotmap import DotMap
import torch
import wandb
import hydra
import omegaconf
from hydra import compose, initialize
from src.common.logger import WandbTrainerLogger
from src.common.train_utils import set_global_seeds
from src.datasets import *
from src.models import *
from src.trainers import *

def run(args):    
    args = DotMap(args)
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides

    # Hydra Compose
    initialize(version_base='1.3', config_path=config_path) 
    cfg = compose(config_name=config_name, overrides=overrides)
    def eval_resolver(s: str):
        return eval(s)
    omegaconf.OmegaConf.register_new_resolver("eval", eval_resolver)
    
    set_global_seeds(seed=cfg.seed)
    device = torch.device(cfg.device)

    # dataloader
    # each subprocess will have a single separate thread.
    torch.set_num_threads(1) 
    train_loaders, test_loader = build_dataloader(cfg.dataset)
    
    # logger
    logger = WandbTrainerLogger(cfg)

    # model
    model = build_model(cfg.model)

    # trainer        
    trainer = build_trainer(cfg=cfg.trainer,
                            device=device,
                            train_loaders=train_loaders,
                            test_loader=test_loader,
                            logger=logger,
                            model=model)

    # run
    for task_idx in range(cfg.num_tasks):
        trainer.train(task_idx)
    
    wandb.finish()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_path', type=str,    default='./configs')
    parser.add_argument('--config_name', type=str,    default='cifar10_resnet18') 
    parser.add_argument('--overrides',   action='append', default=[])
    args = parser.parse_args()

    run(vars(args))
