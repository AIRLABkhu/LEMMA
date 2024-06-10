import os

import numpy as np

import torch
from torch import nn 

from mdistiller.models.memory import Memory


class AugMemory(nn.Module):
    KEYS = ['logits']
    def __init__(self, memory_dir: str, cfg):
        super(AugMemory, self).__init__()
    
        self.memory_dir = memory_dir
    
        self.weak_memory = Memory(self.memory_dir, cfg)
        self.strong_memory = Memory(self.memory_dir, cfg)
        
    @property
    def device(self):
        return self.weak_memory.device

    def reset(self):
        self.weak_memory.reset()
        self.strong_memory.reset()
    
    def forward(self, x, index, **kwargs):
        index = index.cpu()
        weak_logits, weak_features = self.weak_memory.forward(x, index, **kwargs)
        strong_logits, strong_features = self.strong_memory.forward(x, index, **kwargs)
        return [weak_logits, strong_logits], [weak_features, strong_features]
        
    @torch.no_grad()
    def update(self, index, epoch, logits, feature, target, ema_alpha, gamma):
        weak_logits, strong_logits = logits[0], logits[1]
        weak_alpha, strong_alpha = ema_alpha
        weak_gamma, strong_gamma = gamma
        self.weak_memory.update(index, epoch, weak_logits, feature, target, weak_alpha, weak_gamma) # 'feature' would be 'None' because memory only be used when memory distillation
        self.strong_memory.update(index, epoch, strong_logits, feature, target, strong_alpha, strong_gamma)

    @torch.no_grad()
    def export(self, path: str, suffix=None):
        self.weak_memory.export(path=path+'/weak', suffix=suffix)
        self.strong_memory.export(path=path+'/strong', suffix=suffix)
            

if __name__ == '__main__':
    from yacs.config import CfgNode as CN
    from torch import nn
    from .. import cifar as models
    
    cfg = CN()
    cfg.DISTILLER = CN()
    cfg.DISTILLER.TYPE = "REVIEWKD"
    cfg.DISTILLER.TEACHER = "resnet56"
    
    model_type, state_dict_filename = models.cifar_model_dict[cfg.DISTILLER.TEACHER]
    
    memory_type, memory_dir = models.cifar_model_dict[f'{cfg.DISTILLER.TEACHER}_mem']
    model: Memory = memory_type(memory_dir, cfg)
    
    logits, features = model(None, [2, 4, 6])
    print(logits)
    print(len(features['feats']))
    print(features['preact_feats'])
    print(features['pooled_feat'].shape)
    input('Waiting... ')
    