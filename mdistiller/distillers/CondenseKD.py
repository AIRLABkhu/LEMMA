import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from ._base import Distiller
from ._common import normalize, denormalize, adjust_ema_alpha, replicate_logits

def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
    # logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    # logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    logits_student, logits_teacher = logits_student_in, logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class CondenseKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(CondenseKD, self).__init__(student, teacher, cfg)
        self.cfg = cfg
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.attn_loss_weight = cfg.LEMMA.ATTN.LOSS_WEIGHT 
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND 
        self.ema_range = cfg.LEMMA.EMA_RANGE
        self.reset_epochs = set(cfg.LEMMA.RESET)
        self.save_logit = cfg.LEMMA.SAVE_LOGIT
        self.ema_from  = cfg.LEMMA.WARMUP

    def forward_train(self, image, target, index, epoch, phase='full',**kwargs):
        if phase == 'full': # full dataset:
            logits_student, _ = self.student(image)
            if self.logit_stand:
                logits_student = normalize(logits_student)

            loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
            losses_dict = {
                "loss_ce": loss_ce
            }
        
            return logits_student, losses_dict

        elif phase == 'condense':  # condense dataset       
            logits_student, _ = self.student(image)
            with torch.no_grad():
                logits_teacher, _ = self.teacher(image, index)
                
            if self.logit_stand:
                logits_student, _, _ = normalize(logits_student)
                with torch.no_grad():
                    logits_teacher, std_teacher, mean_teacher = normalize(logits_teacher)

            if self.update_teacher:
                logits_attn, ema_alpha = adjust_ema_alpha(self.cfg, epoch, logits_student, logits_teacher, self.attn)
                with torch.no_grad():
                    if epoch in self.reset_epochs:
                        self.teacher.reset()
                    if self.logit_stand:
                        logits_student_may_shift = denormalize(logits_student, std_teacher, mean_teacher)
                    else:
                        logits_student_may_shift = logits_student
                    self.teacher.update(index, epoch, logits_student_may_shift, {}, target, ema_alpha=ema_alpha)
            # Replicas
            if self.cfg.LEMMA.REPLICAS.CARDINALITY > 0:
                cardinality = self.cfg.LEMMA.REPLICAS.CARDINALITY
                noise = self.cfg.LEMMA.REPLICAS.JITTER
                replica_logits_student, replica_logits_teacher = replicate_logits(
                    logits_student, logits_teacher, 
                    cardinality, noise
                )

            if self.cfg.LEMMA.REPLICAS.CARDINALITY > 0:
                loss_kd = self.kd_loss_weight * kd_loss(
                    replica_logits_student, replica_logits_teacher, self.temperature, self.logit_stand
                )
            else:
                loss_kd = self.kd_loss_weight * kd_loss(
                    logits_student, logits_teacher, self.temperature, self.logit_stand
                )
                
            # losses
            loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

            loss_kd = self.kd_loss_weight * kd_loss(
                logits_student, logits_teacher, self.temperature, self.logit_stand
            )
            if logits_attn is not None:
                if epoch >= self.cfg.LEMMA.WARMUP:
                    if self.cfg.LEMMA.ATTN.LOSS_DECAY == "exp":
                        self.attn_loss_weight = min(1, np.exp(- self.cfg.LEMMA.ATTN.LOSS_DECAY_RATIO * (epoch - self.cfg.LEMMA.WARMUP))) * self.attn_loss_weight
                    elif self.cfg.LEMMA.ATTN.LOSS_DECAY == "jump":
                        self.attn_loss_weight = self.cfg.LEMMA.ATTN.LOSS_WEIGHT_JUMP
                loss_attn = self.attn_loss_weight * F.cross_entropy(logits_attn, target)
                losses_dict = {
                    "loss_ce": loss_ce,
                    "loss_kd": loss_kd,
                    "loss_attn": loss_attn,
                }
            else:
                losses_dict = {
                    "loss_ce": loss_ce,
                    "loss_kd": loss_kd,
                }
            
            return logits_student, losses_dict

