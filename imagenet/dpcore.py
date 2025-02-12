from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

from torch.autograd import Variable
from vpt import PromptViT
import numpy as np
import math

class DPCore(nn.Module):
    def __init__(self, model:PromptViT, optimizer, temp_tau=3.0, ema_alpha=0.999, thr_rho=0.8, E_OOD=50):
        super().__init__()
        self.lamda = 1.0
        self.temp_tau = temp_tau
        self.ema_alpha = ema_alpha
        self.thr_rho = thr_rho
        self.E_ID = 1
        self.E_OOD = E_OOD
        

        self.model = model
        self.optimizer = optimizer
        
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

        self.coreset = []


    def _update_coreset(self, weights, batch_mean, batch_std):
        """Update overall test statistics"""
        updated_prompts = self.model.prompts.clone().detach().cpu()
        for p_idx in range(len(self.coreset)):
            self.coreset[p_idx][0] += self.ema_alpha * weights[p_idx] * (batch_mean - self.coreset[p_idx][0])
            self.coreset[p_idx][1] += self.ema_alpha * weights[p_idx] * torch.clamp(batch_std - self.coreset[p_idx][1], min=0.0)
            self.coreset[p_idx][2] += self.ema_alpha * weights[p_idx] * (updated_prompts - self.coreset[p_idx][2])   
            

    @torch.no_grad()
    def _eval_coreset(self, x):
        """Evaluate the coreset on a batch of samples."""
        
        loss, batch_mean, batch_std = forward_and_get_loss(x, self.model, self.lamda, self.train_info, with_prompt=False)
        is_ID = False
        weights = None
        weighted_prompts = None
        if self.coreset:
            weights = calculate_weights(self.coreset, batch_mean, batch_std, self.lamda, self.temp_tau)
            weighted_prompts = torch.stack([w * p[2] for w, p in zip(weights, self.coreset)], dim=0).sum(dim=0)
            assert weighted_prompts.shape == self.model.prompts.shape, f'{weighted_prompts.shape} != {self.model.prompts.shape}'
            self.model.prompts = torch.nn.Parameter(weighted_prompts.cuda())
            self.model.prompts.requires_grad_(False)
            
            loss_new, _, _ = forward_and_get_loss(x, self.model, self.lamda, self.train_info, with_prompt=True)
            if loss_new < loss * self.thr_rho:
                
                self.model.prompts.requires_grad_(True)
                self.optimizer = torch.optim.AdamW([self.model.prompts], lr=1e-1)
                is_ID = True
        else:
            loss_new = loss
            
        return is_ID, batch_mean, batch_std, weighted_prompts, weights, loss, loss_new

    def forward(self, x):
        is_ID, batch_mean, batch_std, weighted_prompts, weights, loss_raw, loss_new = self._eval_coreset(x)
        if is_ID:
            for _ in range(self.E_ID):
                self.model.prompts = torch.nn.Parameter(weighted_prompts.cuda())
                optimizer = torch.optim.AdamW([self.model.prompts], lr=1e-1)
                outputs, loss, batch_mean, batch_std = forward_and_adapt(x, self.model, optimizer, self.lamda, self.train_info)
            self._update_coreset(weights, batch_mean, batch_std)
            
        else:
            
            load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
            self.model.prompts.requires_grad_(True)
            self.optimizer = torch.optim.AdamW([self.model.prompts], lr=1e-1)
            
            for _ in range(self.E_OOD):
                outputs, loss, _, _ = forward_and_adapt(x, self.model, self.optimizer, self.lamda, self.train_info)

            self.coreset.append([batch_mean, batch_std, self.model.prompts.clone().detach().cpu()])
            
        return outputs, loss_raw, loss_new, loss
    
    def obtain_src_stat(self, data_path, num_samples=5000):
        num = 0
        features = []
        import timm
        from torchvision.datasets import ImageNet, STL10
        net = timm.create_model('vit_base_patch16_224', pretrained=True)
        data_config = timm.data.resolve_model_data_config(net)
        src_transforms = timm.data.create_transform(**data_config, is_training=False)
        src_dataset = ImageNet(root=f"{data_path}/ImageNet", split= 'train', transform=src_transforms)
        src_loader = torch.utils.data.DataLoader(src_dataset, batch_size=64, shuffle=True)
        
        with torch.no_grad():
            for _, dl in enumerate(src_loader):
                images = dl[0].cuda()
                feature = self.model.forward_raw_features(images)
                
                output = self.model(images)
                ent = softmax_entropy(output)
                selected_indices = torch.where(ent < math.log(1000)/2-1)[0]
                feature = feature[selected_indices]
                
                features.append(feature[:, 0])
                num += feature.shape[0]
                if num >= num_samples:
                    break

            features = torch.cat(features, dim=0)
            features = features[:num_samples, :]
            print(f'Source Statistics computed with {features.shape[0]} examples.')
            self.train_info = torch.std_mean(features, dim=0)
        del features
        

    def reset(self):
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.coreset = []
        

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

def configure_model(model, cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cfg.TEST.ckpt!=None:
        checkpoint = torch.load(cfg.TEST.ckpt)
        model.load_state_dict(checkpoint, strict=True)
        
    model = PromptViT(model, cfg.OPTIM.PROMPT_NUM)
    model.to(device)
    model.train()
    return model

def collect_params(model):
    return [model.prompts]

# @torch.no_grad()
def forward_and_get_loss(images, model:PromptViT, lamda, train_info, with_prompt=False):
    if with_prompt:
        cls_features = model.forward_features(images)[:, 0]
    else:
        cls_features = model.forward_raw_features(images)[:, 0]
    

    """discrepancy loss"""
    batch_std, batch_mean = torch.std_mean(cls_features, dim=0)
    # std_mse, mean_mse = criterion_mse(batch_std, train_info[0].cuda()), criterion_mse(batch_mean, train_info[1].cuda())
    std_loss = torch.norm(batch_std - train_info[0].cuda(), p=2)
    mean_loss = torch.norm(batch_mean - train_info[1].cuda(), p=2)
    
    loss = lamda * std_loss + mean_loss
    # output = model.vit.forward_head(raw_features)

    return loss, batch_mean, batch_std



def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    
def calculate_weights(coreset, batch_mean, batch_std, lamda, temp_tau):
    mean_tensor = torch.stack([p[0] for p in coreset])
    std_tensor = torch.stack([p[1] for p in coreset])
    assert mean_tensor.shape[1] == 768 and mean_tensor.shape[0] == len(coreset)
    
    mean_match = torch.norm(batch_mean - mean_tensor, p=2, dim=1)
    std_match = torch.norm(batch_std - std_tensor, p=2, dim=1)
    
    match_loss = mean_match + lamda *  std_match
    weights = torch.nn.functional.softmax(-match_loss/temp_tau, dim=0)
    return weights.detach().cpu()

@torch.enable_grad()
def forward_and_adapt(x, model: PromptViT, optimizer, lamda, train_info):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    features = model.forward_features(x)
    cls_features = features[:, 0]
    batch_std, batch_mean = torch.std_mean(cls_features, dim=0)
    # std_mse, mean_mse = criterion_mse(batch_std, train_info[0].cuda()), criterion_mse(batch_mean, train_info[1].cuda())

    std_loss = torch.norm(batch_std - train_info[0].cuda(), p=2)
    mean_loss = torch.norm(batch_mean - train_info[1].cuda(), p=2)
    loss = lamda * std_loss + mean_loss
    
    # output = model.vit.head(cls_features)
    output = model.vit.forward_head(features)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return output, loss, batch_mean, batch_std