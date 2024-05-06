import numpy as np
import tqdm
import copy
import random
import wandb
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from einops import rearrange
from src.common.redo import *
from src.common.schedulers import CosineAnnealingWarmupRestarts
from src.common.metrics import *
from src.common.hessian import *
from src.common.vis_utils import *


class BaseTrainer():
    def __init__(self,
                 cfg,
                 device,
                 train_loaders,
                 test_loader,
                 logger,
                 model):
        
        super().__init__()

        self.cfg = cfg  
        self.device = device
        self.logger = logger
        self.model = model.to(self.device)
        self.init_model = copy.deepcopy(self.model)
        self.target_model = copy.deepcopy(self.model)
        for param in self.target_model.parameters():
            param.requires_grad = False
        
        self.train_loaders = train_loaders
        self.test_loader = test_loader
            
        self.task_idx = 0
        self.epoch = 0
        self.step = 0

    def _build_optimizer(self, optimizer_type, optimizer_cfg):
        if optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), 
                              **optimizer_cfg)
        else:
            raise ValueError

    def _build_scheduler(self, optimizer, scheduler_cfg):
        return CosineAnnealingWarmupRestarts(optimizer=optimizer,
                                             **scheduler_cfg)

    def train(self, task_idx):
        cfg = self.cfg
        
        if (task_idx + 1) > cfg.num_tasks:
            raise ValueError
        self.task_idx = task_idx
        
        train_loader = self.train_loaders[task_idx]
        num_epochs = int(cfg.base_epochs / cfg.input_data_ratios[task_idx])
        test_every = int(cfg.base_test_every / cfg.input_data_ratios[task_idx])        
        
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
        optimizer = self._build_optimizer(cfg.optimizer_type, cfg.optimizer)
        cfg.scheduler.first_cycle_steps = len(train_loader) * num_epochs
        lr_scheduler = self._build_scheduler(optimizer, cfg.scheduler)
                
        test_logs = {}
        test_logs['epoch'] = self.epoch
        if test_every != -1:
            self.model.eval()
            test_logs.update(self.test())
        self.logger.update_log(**test_logs)
        self.logger.log_to_wandb(self.step)
        
        # train        
        for epoch in range(int(num_epochs)):          
            for x_batch, y_batch in tqdm.tqdm(train_loader):   
                # (1) early stop to avoid the loss of plasticity
                # https://link.springer.com/chapter/10.1007/978-3-642-35289-8_5
                max_epoch = int(cfg.early_stop.lmbda * num_epochs)
                if (epoch > max_epoch) and ((task_idx+1) < cfg.num_tasks):
                    train_logs = {}
                    train_logs['lr'] = lr_scheduler.get_lr()[0]
                    
                else:
                    self.model.train()
                    self.target_model.eval()

                    # forward            
                    x_batch = x_batch.to(self.device) # (n, c, h, w)
                    y_batch = y_batch.to(self.device) # (n,)
                    z = self.model.backbone(x_batch)                    
                    y_pred = self.model.head(z)  # (n, y)
                    
                    # nll loss
                    loss_fn = nn.CrossEntropyLoss()
                    nll_loss = loss_fn(y_pred, y_batch)
                    acc1, acc5 = accuracy(y_pred, y_batch, topk=(1, 5))
                    
                    # (2) self-distillation
                    if cfg.self_distill.lmbda > 0:
                        z_t = self.target_model.backbone(x_batch)
                        y_pred_t = self.target_model.head(z_t) # (n,y)
                        soft_logits = F.log_softmax(y_pred / cfg.self_distill.temp, dim=1)
                        soft_target = F.softmax(y_pred_t / cfg.self_distill.temp, dim=1)
                        
                        distill_loss_fn = nn.KLDivLoss()
                        distill_loss = distill_loss_fn(soft_logits, soft_target.detach())    
                    else:
                        distill_loss = torch.zeros(1, device=self.device)          
                    
                    # (3) spectral decoupling loss from gradient starvation
                    # https://arxiv.org/abs/2011.09468
                    spectral_loss = (y_pred ** 2).mean()

                    # (4) regenarative regularization loss
                    # https://arxiv.org/pdf/2308.11958.pdf
                    for param, init_param in zip(
                        self.model.backbone.parameters(), self.init_model.backbone.parameters()):
                        backbone_regen_loss = torch.norm(param - init_param, 2)
                        
                    for param, init_param in zip(
                        self.model.head.parameters(), self.init_model.head.parameters()):
                        head_regen_loss = torch.norm(param - init_param, 2)
                    
                    # loss
                    loss = (nll_loss 
                            + cfg.spectral.lmbda * spectral_loss
                            + cfg.regen.b_lmbda * backbone_regen_loss
                            + cfg.regen.h_lmbda * head_regen_loss
                            + cfg.self_distill.lmbda * distill_loss
                            )

                    # backward
                    scaler.scale(loss).backward()

                    # gradient clipping
                    # unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # EMA (self-distill and Hare and Tortoise)
                    if cfg.ema > 0.0:
                        for slow, fast in zip(self.target_model.parameters(), self.model.parameters()):
                            slow.data = (cfg.ema) * slow.data + (1 - cfg.ema) * fast.data
                            
                        for slow, fast in zip(self.target_model.buffers(), self.model.buffers()):
                            slow.data = (cfg.ema)  * slow.data + (1 - cfg.ema) * fast.data  

                    # log        
                    train_logs = {}                
                    train_logs['train_loss'] = loss.item()
                    train_logs['train_nll_loss'] = nll_loss.item()
                    train_logs['train_spectral_loss'] = spectral_loss.item()
                    train_logs['train_backbone_regen_loss'] = backbone_regen_loss.item()
                    train_logs['train_head_regen_loss'] = head_regen_loss.item()
                    train_logs['train_distill_loss'] = distill_loss.item()
                    train_logs['train_acc1'] = acc1.item()
                    train_logs['train_acc5'] = acc5.item()
                    train_logs['lr'] = lr_scheduler.get_lr()[0]
                    grad_norm_stats = get_grad_norm_stats(self.model)
                    train_logs.update(grad_norm_stats)

                self.logger.update_log(**train_logs)
                if self.step % cfg.log_every == 0:
                    self.logger.log_to_wandb(self.step)
                
                lr_scheduler.step()
                self.step += 1
                    
            self.epoch += 1
            
            # (6) Hare Tortoise
            if (cfg.hare_tortoise.reset_every != -1) and (epoch % cfg.hare_tortoise.reset_every == 0):
                for slow, fast in zip(self.target_model.parameters(), self.model.parameters()):
                    fast.data = slow.data
                    
                for slow, fast in zip(self.target_model.buffers(), self.model.buffers()):
                    fast.data = slow.data

            # log evaluation
            test_logs = {}
            test_logs['epoch'] = self.epoch
            if (self.epoch % test_every == 0) and (test_every != -1):
                self.model.eval()
                self.target_model.eval()
                test_logs.update(self.test())
            self.logger.update_log(**test_logs)
            self.logger.log_to_wandb(self.step)
            
        # (5) Re-Initialization
        # (5.1) Shrink & Perturb: b_lmbda = alpha; h_lmbda = alpha
        # https://arxiv.org/abs/1910.08475
        # (5.2) Head Reset: b_lmbda = 0; h_lmbda = 1.0
        # https://arxiv.org/abs/2205.07802
        for param, init_param in zip(
            self.model.backbone.parameters(), self.init_model.backbone.parameters()):
            param.data = (1-cfg.reinit.b_lmbda) * param.data + cfg.reinit.b_lmbda * init_param.data
            
        for param, init_param in zip(
            self.model.head.parameters(), self.init_model.head.parameters()):
            param.data = (1-cfg.reinit.h_lmbda) * param.data + cfg.reinit.h_lmbda* init_param.data

    
    def test(self) -> dict: 
        cfg = self.cfg
               
        ####################
        ## performance 
        loss_list = []
        acc1_list, acc5_list = [], []      
        acc1_t_list, acc5_t_list = [], []
        N = 0  
        for x_batch, y_batch in tqdm.tqdm(self.test_loader):   
            with torch.no_grad():                        
                x_batch = x_batch.to(self.device) # (n, c, h, w)
                y_batch = y_batch.to(self.device) # (n,)
                z = self.model.backbone(x_batch)
                y_pred = self.model.head(z)
                
                z_t = self.target_model.backbone(x_batch)
                y_pred_t = self.target_model.head(z_t)
                
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(y_pred, y_batch)
                acc1, acc5 = accuracy(y_pred, y_batch, topk=(1, 5))
                acc1_t, acc5_t = accuracy(y_pred_t, y_batch, topk=(1, 5))

            n = len(y_batch)
            loss_list.append(loss * n)
            acc1_list.append(acc1 * n)
            acc5_list.append(acc5 * n)
            acc1_t_list.append(acc1_t * n)
            acc5_t_list.append(acc5_t * n)
            N += n

        loss = torch.sum(torch.stack(loss_list)).item() / N
        acc1 = torch.sum(torch.stack(acc1_list)).item() / N
        acc5 = torch.sum(torch.stack(acc5_list)).item() / N
        acc1_t = torch.sum(torch.stack(acc1_t_list)).item() / N
        acc5_t = torch.sum(torch.stack(acc5_t_list)).item() / N
        
        pred_logs = {
            'test_loss': loss,
            'test_acc1': acc1,
            'test_acc5': acc5,
            'test_target_acc1': acc1_t,
            'test_target_acc5': acc5_t,
        }
        
        ####################
        ## weight analysis
        all_weight_dist, backbone_weight_dist, head_weight_dist = 0.0, 0.0, 0.0
        all_weight_norm, backbone_weight_norm, head_weight_norm = 0.0, 0.0, 0.0

        init_params = {name: param for name, param in self.init_model.named_parameters()}

        for name, param in self.model.named_parameters():
            if name in init_params:
                layer_param_count = param.numel()

                # Compute normalized L2 distance
                weight_dist = torch.norm(param - init_params[name], 2) / layer_param_count
                all_weight_dist += weight_dist.item()

                # Compute normalized L2 norm
                weight_norm = torch.norm(param, 2) / layer_param_count
                all_weight_norm += weight_norm.item()

                # Categorize based on the parameter name
                if 'backbone' in name:
                    backbone_weight_dist += weight_dist.item()
                    backbone_weight_norm += weight_norm.item()
                    
                elif 'head' in name:
                    head_weight_dist += weight_dist.item()
                    head_weight_norm += weight_norm.item()
                    
        weight_logs = {
            'all_weight_dist': all_weight_dist,
            'backbone_weight_dist': backbone_weight_dist,
            'head_weight_dist': head_weight_dist,
            'all_weight_norm': all_weight_norm,
            'backbone_weight_norm': backbone_weight_norm,
            'head_weight_norm': head_weight_norm,
        }
        
        ####################
        # feature analysis        
        # randomly select samples from train / test dataset
        train_dataset = self.train_loaders[self.task_idx].dataset
        n_analysis_samples = cfg.analysis_samples
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            num_workers=0, 
            batch_size=n_analysis_samples, 
            shuffle=True, 
            worker_init_fn=np.random.seed(cfg.seed)
        )

        # obtain features & gradients with given samples
        self.model.enable_hooks()
        x_batch, y_batch = next(iter(train_loader))
        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
        y_pred = self.model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        activations = self.model.get_activations()

        # compute feature metrics
        # zero-activations
        zero_activation_ratios = {}
        for layer_name, activation in activations.items():
            num_zero_activations = torch.sum(activation == 0)
            total_activations = activation.numel()
            zero_ratio = num_zero_activations.float() / total_activations
            zero_activation_ratios[layer_name + '_zero_activ_ratio'] = zero_ratio.item()        

        # feat.rank
        smooth_ranks = {}
        stable_ranks = {}
        
        for layer_name, activation in activations.items():
            if len(activation.shape) == 2:
                # compute singular values
                S = torch.linalg.svdvals(activation)
                smooth_rank, stable_rank = get_rank(S)
                smooth_ranks[layer_name + '_smooth_rank'] = smooth_rank.item()
                stable_ranks[layer_name + '_stable_rank'] = stable_rank.item()

        feature_logs = {}
        feature_logs.update(zero_activation_ratios)
        feature_logs.update(smooth_ranks)
        feature_logs.update(stable_ranks)
        feature_logs = {}

        ####################
        # gradient analysis    
        # zero-grad ratio
        all_zero_grad_ratio, all_param_count = 0.0, 0.0
        backbone_zero_grad_ratio, backbone_param_count = 0.0, 0.0
        head_zero_grad_ratio, head_param_count = 0.0, 0.0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                zero_grad_ratio = torch.sum(param.grad == 0).float() / param.numel()
                all_zero_grad_ratio += zero_grad_ratio * param.numel()
                all_param_count += param.numel()

                if 'backbone' in name:
                    backbone_zero_grad_ratio += zero_grad_ratio * param.numel()
                    backbone_param_count += param.numel()
                elif 'head' in name:
                    head_zero_grad_ratio += zero_grad_ratio * param.numel()
                    head_param_count += param.numel()

        # Calculate the overall average ratios
        all_zero_grad_ratio /= all_param_count
        backbone_zero_grad_ratio /= backbone_param_count
        head_zero_grad_ratio /= head_param_count
        
        grad_logs = {
            'all_zero_grad_ratio': all_zero_grad_ratio,
            'backbone_zero_grad_ratio': backbone_zero_grad_ratio,
            'head_zero_grad_ratio': head_zero_grad_ratio
        }
        
        ######################
        # hessian anlysis
        # use power-iteration method to approximate the eigen / singular values of the hessian.
        if cfg.num_eigen_vals > 0:
            eigen_vals, _ = compute_hessian_eigen_things(
                model = self.model, 
                dataloader = train_loader,
                loss = loss_fn, 
                num_vals = cfg.num_eigen_vals,
                full_dataset = False,
                device = self.device,
            )
            max_eigen_val = eigen_vals[0]
            eigen_vals = torch.from_numpy(eigen_vals.copy())
            
            x_eigen = np.arange(1, len(eigen_vals)+1)
            eigen_spectrum = visualize_plot(
                x=x_eigen, y=eigen_vals, 
                x_label='Index', y_label='Eigenvalue', title='Eigenvalue Spectrum'
            )
            log_eigen_spectrum = visualize_plot(
                x=x_eigen, y=eigen_vals.log(), 
                x_label='Index', y_label='Log Eigenvalue', title='Log Eigenvalue Spectrum'
            )
        else:
            max_eigen_val = 0.0
            dummy_image = PIL.Image.fromarray(np.zeros((3, 3, 3), dtype=np.uint8))
            eigen_spectrum = dummy_image
            log_eigen_spectrum = dummy_image
            
        if cfg.num_singular_vals > 0:
            singular_vals, _, _ = compute_hessian_singular_things(
                model = self.model, 
                dataloader = train_loader,
                loss = loss_fn, 
                num_vals =cfg.num_singular_vals,
                full_dataset = False,
                device = self.device,
            )
            max_singular_val = singular_vals[0]
            singular_vals = torch.from_numpy(singular_vals.copy())
            smooth_hessian_rank, stable_hessian_rank = get_rank(singular_vals)
            
            x_singular = np.arange(1, len(singular_vals)+1)
            singular_spectrum = visualize_plot(
                x=x_singular, y=singular_vals, 
                x_label='Index', y_label='Singular value', title='Singular value Spectrum'
            )
            log_singular_spectrum = visualize_plot(
                x=x_singular, y=singular_vals.log(), 
                x_label='Index', y_label='Log Singular value', title='Log Singular value Spectrum'
            )
        else:
            max_singular_val = 0.0
            smooth_hessian_rank, stable_hessian_rank = 0.0, 0.0
            dummy_image = PIL.Image.fromarray(np.zeros((3, 3, 3), dtype=np.uint8))
            singular_spectrum = dummy_image
            log_singular_spectrum = dummy_image
            

        hessian_logs = {
            'max_eigen_val': max_eigen_val,
            'eigen_spectrum': wandb.Image(eigen_spectrum),
            'log_eigen_spectrum': wandb.Image(log_eigen_spectrum),

            'max_singular_val': max_singular_val,
            'smooth_hessian_rank': smooth_hessian_rank,
            'stable_hessian_rank': stable_hessian_rank,
            'singular_spectrum': wandb.Image(singular_spectrum),
            'log_singular_spectrum': wandb.Image(log_singular_spectrum)         
        }
        
        self.model.disable_hooks()
        
        # (4) ReDO
        # https://arxiv.org/pdf/2302.12902.pdf
        redo_masks = get_redo_masks(activations, cfg.redo.lmbda)
        self.model = reset_dormant_neurons(self.model, redo_masks, use_lecun_init=False)

        test_logs = {}
        test_logs.update(pred_logs)
        test_logs.update(weight_logs)
        test_logs.update(feature_logs)
        test_logs.update(grad_logs)
        test_logs.update(hessian_logs)
        
        return test_logs
