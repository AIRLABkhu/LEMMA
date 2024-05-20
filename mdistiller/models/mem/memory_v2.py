import os

import numpy as np

import torch
from torch import nn

from .stream import MemStream


class MemoryV2(nn.Module):
    def __init__(self, log_dir: str, cfg):
        super(MemoryV2, self).__init__()
        
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
        
        self.use_logits       = bool(mask & 0b1000)
        self.use_feats        = bool(mask & 0b0100)
        self.use_preact_feats = bool(mask & 0b0010)
        self.use_pooled_feat  = bool(mask & 0b0001)
        
        mem_name = cfg.DISTILLER.TEACHER
        num_workers = cfg.LEMMA.NUM_WORKERS
        mem_dir = cfg.LEMMA.MEM_DIR
        
        
        
        self.dummy = nn.Parameter(torch.zeros(0))
        
    @property
    def device(self):
        return self.dummy.device
    
    def forward(self, x, index, **kwargs):
        def tensor_from_arr(arr):
            return torch.from_numpy(np.stack(arr, axis=0)).to(x.device)
        
        index = index.cpu().tolist()
        logits = None if self.use_logits else \
            tensor_from_arr(self.stream.read(map('logits/{:08d}.npy'.format, index)))
            
        features = {
            'feats': None if self.use_feats else \
                [tensor_from_arr(f) for f in 
                 zip(self.stream.read(map('feats/{:08d}.npy'.format, index)))],
                
            'preact_feats': None if self.use_preact_feats else \
                [tensor_from_arr(f) for f in 
                 zip(self.stream.read(map('preact_feats/{:08d}.npy'.format, index)))],
                
            'pooled_feat': None if self.use_pooled_feat else \
                tensor_from_arr(self.stream.read(map('pooled_feat/{:08d}.npy'.format, index))),
        }
        return logits, features
        
    @torch.no_grad()
    def update(self, index, student_logits, student_feature, teacher_logits, teacher_feature, ema_alpha: float):
        index = index.cpu()
        ema = ema_alpha
        _ema = 1 - ema  # .........| for student 
        
        if self.use_logits:
            logits = (student_logits * _ema + teacher_logits * ema).cpu().numpy() 
            
        if self.use_feats:
            student_feats = student_feature['feats'] if 'feats' in student_feature else None 
            teacher_feats = teacher_feature['feats'] if 'feats' in teacher_feature else None 
            feats = [(s * _ema + t * ema).cpu().numpy() for s, t in zip(student_feats, teacher_feats)] 

        if self.use_preact_feats:
            student_preact_feats = student_feature['preact_feats'] if 'preact_feats' in student_feature else None 
            teacher_preact_feats = teacher_feature['preact_feats'] if 'preact_feats' in teacher_feature else None 
            preact_feats = [(s * _ema + t * ema).cpu().numpy() for s, t in zip(student_preact_feats, teacher_preact_feats)] 

        if self.use_pooled_feat:
            student_pooled_feat = student_feature['pooled_feat'] if 'pooled_feat' in student_feature else None 
            teacher_pooled_feat = teacher_feature['pooled_feat'] if 'pooled_feat' in teacher_feature else None 
            pooled_feat = (student_pooled_feat * _ema + teacher_pooled_feat * ema).cpu().numpy() 
    