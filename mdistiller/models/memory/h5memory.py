import os
import shutil
import h5py

import numpy as np

import torch
from torch import nn 


class H5Memory(nn.Module):
    KEYS = ['logits', 'feats', 'preact_feats', 'pooled_feat']
    
    def __init__(self, memory_dir: str, cfg):
        super(H5Memory, self).__init__()
        
        mask = {
            "NONE"      : 0b0000, 
            "KD"        : 0b1000, 
            "MLKD"      : 0b1000, 
            "RKD"       : 0b0001, 
            "CRD"       : 0b0001, 
            "PKT"       : 0b0001, 
            "Sonly"     : 0b1000, 
            "DKD"       : 0b1000, 
        }[cfg.DISTILLER.TYPE]
        
        self.origin_filename = os.path.join(memory_dir, 'memory.hdf5')
        if not os.path.exists(self.origin_filename):
            raise FileNotFoundError(f"Memory not found: '{self.origin_filename}'.")
        
        self.working_filename = None
        self.use_logits       = bool(mask & 0b1000)
        self.use_pooled_feat  = bool(mask & 0b0001)
        
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
        
        self.__x_lower = cfg.LEMMA.WARMUP
        self.__x_upper = cfg.SOLVER.EPOCHS
        self.__ema = cfg.LEMMA.EMA_RANGE
        self.ema_step = cfg.LEMMA.EMA_STEP
        
        self.h5 = None
        self.dummy = nn.Parameter(torch.zeros(0))
        
    def __del__(self):
        self.h5.close()
        
    @property
    def device(self):
        return self.dummy.device
        
    def initdir(self, logdir, *suffix):
        filename = '_'.join(['memory', *suffix])
        self.working_filename = os.path.join(logdir, f'{filename}.hdf5')
        shutil.copyfile(self.origin_filename, self.working_filename)
        self.h5 = h5py.File(self.working_filename, 'a')
    
    def reset(self):
        with h5py.File(self.origin_filename, 'r') as origin:
            self.h5['logits'][:] = origin['logits']
            self.h5['pooled_feat'][:] = origin['pooled_feat']
            
    def _sort_index(index):
        if torch.is_tensor(index):
            index = index.cpu().numpy()
        sorted_index = np.sort(index)
        index_inv = np.argsort(index)
        index_iinv = np.argsort(index_inv)
        return sorted_index, index_inv, index_iinv
    
    def forward(self, x, index, **kwargs):
        index = index.cpu().numpy()
        sorted_index, _, index_iinv = H5Memory._sort_index(index)
        return (
            torch.from_numpy(self.h5['logits'][sorted_index])[index_iinv].to(self.device), 
            {
                'pooled_feat': torch.from_numpy(self.h5['pooled_feat'][sorted_index][index_iinv]).to(self.device),
            }
        )
        
    @torch.no_grad()
    def update(self, index, epoch, logits, feature, target, ema_alpha):
        if (epoch < self.__x_lower) or (self.ema_stop <= epoch):
            return
        if (epoch - self.__x_lower + 1) % self.ema_step != 0:
            return
        if self.__ema is None:
            return 
        
        index = index.cpu()
        sorted_index, index_inv, index_iinv = H5Memory._sort_index(index)
        pooled_feat = feature['pooled_feat'] if 'pooled_feat' in feature else None
    
        if isinstance(ema_alpha, torch.Tensor):
            ema_alpha = ema_alpha.cpu().unsqueeze(-1)
        else:
            ema_alpha = torch.ones(index.size(0), 1, dtype=torch.float) * ema_alpha
        ema = ema_alpha
        _ema = 1 - ema  # .........| for student
        
        if (logits is not None) and self.use_logits:
            mem_logits = torch.from_numpy(self.h5['logits'][sorted_index])[index_iinv]
            grad = _ema * (mem_logits - logits.cpu())
            mem_logits -= grad

            if self.use_logit_aug and (self.__x_lower < epoch) and (epoch < self.logit_aug_stop):
                if epoch == self.__x_lower + 1: 
                    preds = np.argmax(self.h5['logits'], axis=1)
                    self.h5['logits'].attrs['centroids'] = np.stack([
                        self.h5['logits'][preds == i].mean(axis=0) for i in range(self.num_classes)
                    ], axis=0) 
                    self.h5['logits'].attrs['num_samples'] = [
                        int((preds == i).sum()) for i in range(self.num_classes)
                    ]
                    self.logit_centroids = self.h5['logits'].attrs['centroids']
                    self.num_samples = self.h5['logits'].attrs['num_samples']

                target_uniques = target.unique().tolist()
                for t in target_uniques:
                    t_mask = (target == t).cpu()
                    self.logit_centroids[t] = (self.logit_centroids[t] * self.num_samples[t] + logits[t_mask].sum(dim=0).cpu().numpy()) / (self.num_samples[t] + t_mask.sum().numpy())
                    self.num_samples[t] += t_mask.sum().item()
                    
                beta = self.adjust_logit_aug_beta(epoch, logits, target, index)
                centroids = torch.from_numpy(self.logit_centroids[target.cpu()])
                mem_logits = (centroids * beta) + (mem_logits * (1-beta))
                
            if self.logit_aug_kwargs.NOISE:
                mem_logits = mem_logits + torch.normal(mean=0, std=self.logit_aug_kwargs.NOISE, size=mem_logits.shape)
                
            self.h5['logits'][sorted_index] = mem_logits[index_inv].numpy()
                        
        if (pooled_feat is not None) and self.use_pooled_feat:
            mem_pooled_feat = self.h5['pooled_feat'][index]
            mem_pooled_feat = (pooled_feat.cpu() * _ema) + (mem_pooled_feat * ema)
            self.pooled_feat[index] = mem_pooled_feat.numpy()

    def adjust_logit_aug_beta(self, epoch, logits, target, index, eps=1.0E-8):
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
                
                ce = -nn.functional.cross_entropy(l1_samples, l1_targets, reduction='none')  # ..........| bs * n[0]
                ce = ce.reshape(batch_size, num_samples).cpu()
                # weights = ce - ce.min(dim=1, keepdim=True).values
                # weights = weights / (weights.max(dim=1, keepdim=True).values + eps)  # ..................| bs, n[0] -> minmax 
                weights = torch.softmax(ce, dim=1)
                
                if transmit:
                    t = weights.cumsum(dim=1)
                    weights = torch.exp(-t) * weights
                
                return (weights * torch.arange(num_samples)[None]).mean(dim=1, keepdim=True)  # .........| bs, 1 

    @torch.no_grad()
    def export(self, path: str, suffix=None):
        if self.h5 is None:
            return
        if suffix is not None:
            path = f'{path}_{suffix}'
        
        history = self.h5.require_group('history')
        history.attrs['use_logits'] = self.use_logits
        history.attrs['use_pooled_feat'] = self.use_pooled_feat
        
        group = history.create_group(path)
        if self.use_logits:
            group.create_dataset('logit', data=self.h5['logits'][:])
        if self.use_pooled_feat:
            group.create_dataset('pooled_feat', data=self.h5['pooled_feat'][:])
            

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
    model: H5Memory = memory_type(memory_dir, cfg)
    
    logits, features = model(None, [2, 4, 6])
    print(logits)
    print(len(features['feats']))
    print(features['preact_feats'])
    print(features['pooled_feat'].shape)
    input('Waiting... ')
    