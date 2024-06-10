import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import normalize, denormalize, adjust_ema_alpha

def dkd_loss(logits_student_in, logits_teacher_in, target, alpha, beta, temperature, logit_stand):
    # logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    # logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    
    logits_student, logits_teacher = logits_student_in, logits_teacher_in

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class DKD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(DKD, self).__init__(student, teacher, cfg)
        self.cfg = cfg
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND 
        # LEMMA
        self.attn_loss_weight = cfg.LEMMA.ATTN.LOSS_WEIGHT
        self.ema_range = cfg.LEMMA.EMA_RANGE
        self.ema_from  = cfg.LEMMA.WARMUP
        self.save_logit = cfg.LEMMA.SAVE_LOGIT

    def forward_train(self, image, target, index, epoch, **kwargs):
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
                gamma = 2 * ema_alpha * (ema_alpha - 1)  + 1
                logits_student_may_shift = logits_student / gamma.unsqueeze(-1)
                logits_student_may_shift = denormalize(logits_student_may_shift, std_teacher, mean_teacher)
                self.teacher.update(index, epoch, logits_student_may_shift, {}, target, ema_alpha=ema_alpha, gamma=gamma)
        else:
            logits_attn = None

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        loss_dkd = min(epoch / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
            self.logit_stand,
        )
        if logits_attn is not None:
            loss_attn = self.attn_loss_weight * F.cross_entropy(logits_attn, target)
            losses_dict = {
                "loss_ce": loss_ce,
                "loss_kd": loss_dkd,
                "loss_attn" : loss_attn,
            }
        else:
            losses_dict = {
                "loss_ce": loss_ce,
                "loss_kd": loss_dkd,
            }
        return logits_student, losses_dict
