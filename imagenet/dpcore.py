"""
Copyright to FOA Authors ICML 2024
"""

from argparse import ArgumentDefaultsHelpFormatter
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

from torch.autograd import Variable
from vpt import PromptViT
import numpy as np
import time
import math

# from utils.cli_utils import accuracy, AverageMeter

RUNNING_IMAGNET_R = False

class DPCore(nn.Module):
    """test-time Forward Optimization Adaptation
    FOA devises both input level and output level adaptation.
    It avoids modification to model weights and adapts in a backpropogation-free manner.
    """
    def __init__(self, model:PromptViT, tau=3.0, alpha=0.999, rho=0.8, E_ID=1, E_OOD=100):
        super().__init__()
        self.lamda = 1.0
        self.tau = tau
        self.alpha = alpha
        self.rho = rho
        self.E_ID = E_ID
        self.E_OOD = E_OOD
        

        self.model = model
        self.optimizer = torch.optim.AdamW([self.model.prompts], lr=1e-1, weight_decay=1e-5)
        
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

        self.coreset = []


    def _update_coreset(self, weights, batch_mean, batch_std):
        """Update overall test statistics, Eqn. (9)"""
        updated_prompts = self.model.prompts.clone().detach().cpu()
        for p_idx in range(len(self.coreset)):
            self.coreset[p_idx][0] += self.alpha * weights[p_idx] * (batch_mean - self.coreset[p_idx][0])
            self.coreset[p_idx][1] += self.alpha * weights[p_idx] * torch.clamp(batch_std - self.coreset[p_idx][1], min=0.0)
            self.coreset[p_idx][2] += self.alpha * weights[p_idx] * (updated_prompts - self.coreset[p_idx][2])
            # self.coreset[p_idx][0] += self.alpha  * (batch_mean - self.coreset[p_idx][0])
            # self.coreset[p_idx][1] += self.alpha  * torch.clamp(batch_std - self.coreset[p_idx][1], min=0.0)
            # self.coreset[p_idx][2] += self.alpha  * (updated_prompts - self.coreset[p_idx][2])   
            

    @torch.no_grad()
    def _eval_coreset(self, x):
        """Evaluate the coreset on a batch of samples."""
        
        loss, batch_mean, batch_std = forward_and_get_loss(x, self.model, self.lamda, self.train_info, with_prompt=False)
        is_ID = False
        weights = None
        weighted_prompts = None
        if self.coreset:
            weights = calculate_weights(self.coreset, batch_mean, batch_std, self.lamda, self.tau)
            weighted_prompts = torch.stack([w * p[2] for w, p in zip(weights, self.coreset)], dim=0).sum(dim=0)
            assert weighted_prompts.shape == self.model.prompts.shape, f'{weighted_prompts.shape} != {self.model.prompts.shape}'
            self.model.prompts = torch.nn.Parameter(weighted_prompts.cuda())
            self.model.prompts.requires_grad_(False)
            
            loss_new, _, _ = forward_and_get_loss(x, self.model, self.lamda, self.train_info, with_prompt=True)
            # print(f'EVAL: {loss.item()} -> {loss_new.item()}')
            if loss_new < loss * self.rho:
                
                self.model.prompts.requires_grad_(True)
                self.optimizer = torch.optim.AdamW([self.model.prompts], lr=1e-1, weight_decay=1e-5)
                is_ID = True
        else:
            loss_new = loss
            
        return is_ID, batch_mean, batch_std, weighted_prompts, weights, loss, loss_new

    def forward(self, x):
        is_ID, batch_mean, batch_std, weighted_prompts, weights, loss_raw, loss_new = self._eval_coreset(x)
        if is_ID:
            for _ in range(self.E_ID):
                self.model.prompts = torch.nn.Parameter(weighted_prompts.cuda())
                optimizer = torch.optim.AdamW([self.model.prompts], lr=1e-1, weight_decay=1e-5)
                outputs, loss, batch_mean, batch_std = forward_and_adapt(x, self.model, optimizer, self.lamda, self.train_info)
            self._update_coreset(weights, batch_mean, batch_std)
            
        else:
            
            load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
            self.model.prompts.requires_grad_(True)
            self.optimizer = torch.optim.AdamW([self.model.prompts], lr=1e-1, weight_decay=1e-5)
            
            for _ in range(self.E_OOD):
                outputs, loss, _, _ = forward_and_adapt(x, self.model, self.optimizer, self.lamda, self.train_info)
                # print(f'OOD: {loss.item()}', end=' ')
            self.coreset.append([batch_mean, batch_std, self.model.prompts.clone().detach().cpu()])
        return outputs, loss_raw, loss_new, loss
    
    def obtain_src_stat(self, train_loader=None):
        print('===> begin calculating mean and variance')
        # features = []
        # with torch.no_grad():
        #     for _, dl in enumerate(train_loader):
        #         images = dl[0].cuda()
        #         feature = self.model.forward_raw_features(images)
        #         features.append(feature[:, 0])
        #         # break
        #     features = torch.cat(features, dim=0)
        #     self.train_info = torch.std_mean(features, dim=0)
        # del features
        
        self.train_info = torch.load('./train_info.pth')
        # torch.save(self.train_info, '/media/zybeich/FOA/train_info.pth')
        print('===> calculating mean and variance end')

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

# criterion_mse = nn.MSELoss(reduction='none').cuda()
# criterion_mse = nn.MSELoss(reduction='mean').cuda()

# @torch.no_grad()
def forward_and_get_loss(images, model:PromptViT, lamda, train_info, with_prompt=False):
    if with_prompt:
        cls_features = model.forward_features(images)[:, 0]
    else:
        cls_features = model.forward_raw_features(images)[:, 0]
    

    """discrepancy loss for Eqn. (5)"""
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
    
def calculate_weights(coreset, batch_mean, batch_std, lamda, tau):
    mean_tensor = torch.stack([p[0] for p in coreset])
    std_tensor = torch.stack([p[1] for p in coreset])
    assert mean_tensor.shape[1] == 768 and mean_tensor.shape[0] == len(coreset)
    
    mean_match = torch.norm(batch_mean - mean_tensor, p=2, dim=1)
    std_match = torch.norm(batch_std - std_tensor, p=2, dim=1)
    
    match_loss = mean_match + lamda *  std_match
    weights = torch.nn.functional.softmax(-match_loss/tau, dim=0)
    # weights = weights.unsqueeze(-1).unsqueeze(-1)
    # print(f'weights: {weights}, sum: {weights.sum().item()}, loss: {match_loss}')
    # print(f'weights: {weights.tolist()}')
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