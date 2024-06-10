import os

import numpy as np

import torch
from torch import nn 


class Memory(nn.Module):
    KEYS = ['logits', 'feats', 'preact_feats', 'pooled_feat']
    
    def __init__(self, memory_dir: str, cfg):
        super(Memory, self).__init__()
        
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
        
        self.memory_dir = memory_dir
        self.use_logits       = bool(mask & 0b1000)
        self.use_feats        = bool(mask & 0b0100)
        self.use_preact_feats = bool(mask & 0b0010)
        self.use_pooled_feat  = bool(mask & 0b0001)

        self.logits       = self._load(Memory.KEYS[0]) if self.use_logits       else None
        self.feats        = self._load(Memory.KEYS[1]) if self.use_feats        else None
        self.preact_feats = self._load(Memory.KEYS[2]) if self.use_preact_feats else None
        self.pooled_feat  = self._load(Memory.KEYS[3]) if self.use_pooled_feat  else None
        
        self.ema_stop = cfg.LEMMA.STOP
        if self.ema_stop == -1:
            self.ema_stop = np.inf
        
        self.logit_aug_kwargs = cfg.LEMMA.LOGIT_AUG
        self.use_logit_aug = self.logit_aug_kwargs.ENABLE
        self.num_classes = {'cifar100': 100, 'imagenet': 1000}[cfg.DATASET.TYPE]
        self.logit_centroids = None
        self.num_samples = None
        self.logit_aug_stop = self.logit_aug_kwargs.STOP
        if self.logit_aug_stop == -1:
            self.logit_aug_stop = np.inf

        self.use_adam = cfg.LEMMA.ADAM.ENABLE
        self.adam_t = torch.zeros_like(self.logits) if self.use_adam else None
        self.adam_m = torch.zeros_like(self.logits) if self.use_adam else None
        self.adam_v = torch.zeros_like(self.logits) if self.use_adam else None
        self.adam_hparams = cfg.LEMMA.ADAM
        
        self.__x_lower = cfg.LEMMA.WARMUP
        self.__x_upper = cfg.SOLVER.EPOCHS
        self.__ema = cfg.LEMMA.EMA_RANGE
        self.ema_step = cfg.LEMMA.EMA_STEP
        
        self.dummy = nn.Parameter(torch.zeros(0))
        
    @property
    def device(self):
        return self.dummy.device
        
    def _load(self, target):
        path = os.path.join(self.memory_dir, f'{target}.pth')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Memory not found: '{path}'.")
        
        return torch.load(path, map_location='cpu')
    
    def reset(self):
        self.logits       = self._load(Memory.KEYS[0]) if self.use_logits       else None
        self.feats        = self._load(Memory.KEYS[1]) if self.use_feats        else None
        self.preact_feats = self._load(Memory.KEYS[2]) if self.use_preact_feats else None
        self.pooled_feat  = self._load(Memory.KEYS[3]) if self.use_pooled_feat  else None
    
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
    def update(self, index, epoch, logits, feature, target, ema_alpha, gamma):
        if (epoch < self.__x_lower) or (self.ema_stop <= epoch):
            return
        if (epoch - self.__x_lower + 1) % self.ema_step != 0:
            return
        if self.__ema is None:
            return 
        
        index = index.cpu()
        feats = feature['feats'] if 'feats' in feature else None
        preact_feats = feature['preact_feats'] if 'preact_feats' in feature else None
        pooled_feat = feature['pooled_feat'] if 'pooled_feat' in feature else None
    
        if isinstance(ema_alpha, torch.Tensor):
            ema_alpha = ema_alpha.cpu().unsqueeze(-1)
        else:
            ema_alpha = torch.ones(index.size(0), 1, dtype=torch.float) * ema_alpha
        ema = ema_alpha
        _ema = 1 - ema  # .........| for student
        
        if (logits is not None) and self.use_logits:
            self.logits[index] = self.logits[index] / gamma.unsqueeze(-1).cpu()
            grad = _ema * (self.logits[index] - logits.cpu())
            
            if self.use_adam:
                betas = self.adam_hparams.BETAS
                eps = self.adam_hparams.EPS
                self.adam_t[index] += 1
                self.adam_m[index] = ((betas[0] * self.adam_m[index].cuda()) + ((1 - betas[0]) * grad.cuda())).cpu()
                self.adam_v[index] = ((betas[1] * self.adam_v[index].cuda()) + ((1 - betas[1]) * torch.square(grad.cuda()))).cpu()
                m_hat = self.adam_m[index] / (1 - betas[0]**self.adam_t[index])
                v_hat = self.adam_v[index] / (1 - betas[1]**self.adam_t[index])
                grad = m_hat / (torch.sqrt(v_hat) + eps)
            self.logits[index] -= grad

            if self.use_logit_aug and (self.__x_lower < epoch) and (epoch < self.logit_aug_stop):
                if self.logit_centroids is None:
                    preds = self.logits.argmax(dim=1)
                    self.logit_centroids = torch.stack([
                        self.logits[preds == i].mean(dim=0) for i in range(self.num_classes)
                    ], dim=0) 
                    self.num_samples = [
                        (preds == i).sum().item() for i in range(self.num_classes)
                    ]

                target_uniques = target.unique().tolist()
                for t in target_uniques:
                    t_mask = (target == t).cpu()
                    self.logit_centroids[t] = (self.logit_centroids[t] * self.num_samples[t] + logits[t_mask].sum(dim=0).cpu()) / (self.num_samples[t] + t_mask.sum())
                    self.num_samples[t] += t_mask.sum().item()
                    
                beta = self.adjust_logit_aug_beta(epoch, logits, target, index)
                centroids = self.logit_centroids[target.cpu()]
                self.logits[index] = (centroids * beta) + (self.logits[index] * (1-beta))
                
            if self.logit_aug_kwargs.NOISE:
                self.logits[index] = self.logits[index] + torch.normal(mean=0, std=self.logit_aug_kwargs.NOISE, size=self.logits[index].shape)
                        
        if (feats is not None) and self.use_feats:
            for i in range(len(feats)):
                self.feats[i][index] = (feats[i].cpu() * _ema) + (self.feats[i][index] * ema)
        if (preact_feats is not None) and self.use_preact_feats:
            for i in range(len(preact_feats)):
                self.preact_feats[i][index] = (preact_feats[i].cpu() * _ema) + (self.preact_feats[i][index] * ema)
        if (pooled_feat is not None) and self.use_pooled_feat:
            self.pooled_feat[index] = (pooled_feat.cpu() * _ema) + (self.pooled_feat[index] * ema)

    def adjust_logit_aug_beta(self, epoch, logits, target, index):
        match self.logit_aug_kwargs.STRATEGY:
            case 'const-rand':
                beta = self.logit_aug_kwargs.RANGE[0] 
                beta = torch.rand_like(logits, device='cpu') * beta
                return beta
            case 'lin-rand':
                aug_range = self.logit_aug_kwargs.RANGE
                beta = (epoch - self.__x_lower) / (self.__x_upper - self.__x_lower)
                beta = ((1 - beta) * aug_range[0]) + (beta *  aug_range[1])
                beta = torch.rand_like(logits, device='cpu') * beta
                return beta
            case 'attn':
                device = logits.device
                batch_size = logits.size(0)
                num_samples = self.logit_aug_kwargs.ATTN 
                transmit = self.logit_aug_kwargs.TRANSMIT 
                
                steps = torch.linspace(0, 1, num_samples + 2)[None, 1:-1, None].to(device)  # ...........| 1, n[0], 1 
                centroids = self.logit_centroids[target.cpu()][:, None]  # ..............................| bs, 1, cls 
                logits = logits[:, None]  # .............................................................| bs, 1, cls 
                
                l1_samples = ((centroids.to(device) * steps) + (logits * (1-steps))).flatten(0, 1)  # ...| bs * n[0], cls 
                l1_targets = target[:, None].repeat(1, num_samples).flatten(0, 1)  # ....................| bs * n[0] 
                
                ce = -nn.functional.cross_entropy(l1_samples, l1_targets, reduction='none')  # .....| bs * n[0]
                ce = ce.reshape(batch_size, num_samples).cpu()
                weights = ce - ce.min(dim=1, keepdim=True).values
                weights = weights / weights.max(dim=1, keepdim=True).values  # .................| bs, n[0] -> minmax 
                
                if transmit:
                    t = weights.cumsum(dim=1)
                    weights = torch.exp(-t) * weights
                
                return (weights * torch.arange(num_samples)[None]).mean(dim=1, keepdim=True)  # ......| bs, 1 

    @torch.no_grad()
    def export(self, path: str, suffix=None):
        from pathlib import Path
        path: Path = Path(path).joinpath(suffix)
        path.mkdir(parents=True, exist_ok=True)
        if self.logits is not None:
            np.save(str(path.joinpath('logits.npy')), self.logits.numpy())
        if self.feats is not None:
            torch.save(self.feats, str(path.joinpath('feats.pt')))
        if self.preact_feats is not None:
            torch.save(self.preact_feats, str(path.joinpath('preact_feats.pt')))
        if self.pooled_feat is not None:
            np.save(str(path.joinpath('pooled_feat.npy')), self.pooled_feat.numpy())
            

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
    