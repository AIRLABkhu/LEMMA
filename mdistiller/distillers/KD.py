import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import normalize, denormalize, adjust_ema_alpha

def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
    # logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    # logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    logits_student, logits_teacher = logits_student_in, logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher, cfg)
        self.cfg = cfg
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND 
        self.ema_range = cfg.LEMMA.EMA_RANGE

    def forward_train(self, image, target, index, epoch, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image, index)
            
        if self.logit_stand:
            logits_student, _, _ = normalize(logits_student)
            with torch.no_grad():
                logits_teacher, std_teacher, mean_teacher = normalize(logits_teacher)
            
        with torch.no_grad():
            if self.update_teacher:
                ema_alpha = adjust_ema_alpha(self.cfg, epoch, logits_student, logits_teacher, None)
                logits_student_may_shift = denormalize(logits_student, std_teacher, mean_teacher) if self.logit_stand else logits_student
                self.teacher.update(index, epoch, logits_student_may_shift, {}, ema_alpha=ema_alpha)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature, self.logit_stand
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
