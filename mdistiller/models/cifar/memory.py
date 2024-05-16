import os

import numpy as np

import torch
from torch import nn
    

class Memory(nn.Module):
    def __init__(self, memory_dir: str, cfg):
        super(Memory, self).__init__()
        
        def load(target):
            path = os.path.join(memory_dir, f'{target}.pth')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Memory not found: '{path}'.")
            
            return torch.load(path, map_location='cpu')
        
        mask = {
            "NONE"      : 0b0000, 
            "KD"        : 0b1000, 
            "MLKD"      : 0b1000, 
            "AT"        : 0b0100, 
            "OFD"       : 0b0010, 
            "RKD"       : 0b0001, 
            "FITNET"    : 0b0100, 
            "KDSVD"     : 0b0100, 
            "CRD"       : 0b0001, 
            "NST"       : 0b0100, 
            "PKT"       : 0b0001, 
            "SP"        : 0b0100, 
            "Sonly"     : 0b1000, 
            "VID"       : 0b0100, 
            "REVIEWKD"  : 0b0101, 
            "DKD"       : 0b1000, 
        }[cfg.DISTILLER.TYPE]
        keys = ['logits', 'feats', 'preact_feats', 'pooled_feat']
        
        self.use_logits       = bool(mask & 0b1000)
        self.use_feats        = bool(mask & 0b0100)
        self.use_preact_feats = bool(mask & 0b0010)
        self.use_pooled_feat  = bool(mask & 0b0001)

        self.logits       = load(keys[0]) if self.use_logits       else None
        self.feats        = load(keys[1]) if self.use_feats        else None
        self.preact_feats = load(keys[2]) if self.use_preact_feats else None
        self.pooled_feat  = load(keys[3]) if self.use_pooled_feat  else None
        
        self.__x_lower = cfg.DISTILLER.EMA_FROM
        self.__x_upper = cfg.SOLVER.EPOCHS
        self.__ema = cfg.DISTILLER.EMA
        self.__y_upper, self.__y_lower = self.__ema
        self.__gamma = cfg.DISTILLER.EMA_GAMMA
        self.__ema_scheduler = cfg.DISTILLER.EMA_SCHEDULER
        
        self.dummy = nn.Parameter(torch.zeros(0))
        
    @property
    def device(self):
        return self.dummy.device
    
    def forward(self, x, index, **kwargs):
        index = index.cpu()
        logits = None if self.logits is None else self.logits[index].to(x.device)
        features = {
            'feats': None if self.feats is None else [f[index].to(x.device) for f in self.feats],
            'preact_feats': None if self.preact_feats is None else [f[index].to(x.device) for f in self.preact_feats],
            'pooled_feat': None if self.pooled_feat is None else self.pooled_feat[index].to(x.device),
        }
        return logits, features
        
    @torch.no_grad()
    def update(self, index, epoch, logits, feature, ema_ratio):
        if epoch < self.__x_lower:
            return
        if self.__ema is None:
            return 
        if ema_ratio == 1.0:
            print("not updating now")
            return
        
        index = index.cpu()
        feats = feature['feats'] if 'feats' in feature else None
        preact_feats = feature['preact_feats'] if 'preact_feats' in feature else None
        pooled_feat = feature['pooled_feat'] if 'pooled_feat' in feature else None

        if self.__ema_scheduler is "Linear":
            ema = (epoch - self.__x_lower) / (self.__x_upper - self.__x_lower)
            ema = np.power(1 - np.power(ema, self.__gamma), 1 / self.__gamma)
            ema_ratio = ema_ratio * (self.__y_upper - self.__y_lower) + self.__y_lower
        ema = ema_ratio
        _ema = 1 - ema  # .........| for student
        
        if (logits is not None) and self.use_logits:
            self.logits[index] = (logits.cpu() * _ema) + (self.logits[index] * ema)
        if (feats is not None) and self.use_feats:
            for i in range(len(feats)):
                self.feats[i][index] = (feats[i].cpu() * _ema) + (self.feats[i][index] * ema)
        if (preact_feats is not None) and self.use_preact_feats:
            for i in range(len(preact_feats)):
                self.preact_feats[i][index] = (preact_feats[i].cpu() * _ema) + (self.preact_feats[i][index] * ema)
        if (pooled_feat is not None) and self.use_pooled_feat:
            self.pooled_feat[index] = (pooled_feat.cpu() * _ema) + (self.pooled_feat[index] * ema)
            

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
    