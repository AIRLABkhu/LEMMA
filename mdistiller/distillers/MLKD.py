from termios import CEOL
from turtle import st
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ._base import Distiller
from .loss import CrossEntropyLabelSmooth
from ._common import normalize, denormalize, adjust_ema_alpha

def kd_loss(logits_student_in, logits_teacher_in, temperature, reduce=True, logit_stand=False):
    # logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    # logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    logits_student, logits_teacher = logits_student_in, logits_teacher_in

    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    if reduce:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    else:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature**2
    return loss_kd


def cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss


def bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / batch_size
    return consistency_loss


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_data_conf(x, y, lam, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = lam.reshape(-1,1,1,1)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class MLKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(MLKD, self).__init__(student, teacher, cfg)
        self.cfg = cfg
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND 
        # LEMMA
        self.attn_loss_weight = cfg.LEMMA.ATTN.LOSS_WEIGHT
        self.ema_range = cfg.LEMMA.EMA_RANGE
        self.ema_from  = cfg.LEMMA.WARMUP
        self.save_logit = cfg.LEMMA.SAVE_LOGIT

    def forward_train(self, image_weak, image_strong, target, index, epoch, **kwargs):
        logits_student_weak, _ = self.student(image_weak)
        logits_student_strong, _ = self.student(image_strong)
        with torch.no_grad():
            if self.update_teacher:
                logits_ , _ = self.teacher(image_weak, index)
                logits_teacher_weak, logits_teacher_strong = logits_[0], logits_[1]
            else:
                logits_teacher_weak, _ = self.teacher(image_weak, index)
                logits_teacher_strong, _ = self.teacher(image_strong, index)
     
        logits_student_weak_may_stand, _, _ = normalize(logits_student_weak) if self.logit_stand else logits_student_weak
        logits_student_strong_may_stand, _, _ = normalize(logits_student_strong) if self.logit_stand else logits_student_strong
        with torch.no_grad():
            logits_teacher_weak_may_stand, std_teacher_weak, mean_teacher_weak = normalize(logits_teacher_weak) if self.logit_stand else logits_teacher_weak
            logits_teacher_strong_may_stand, std_teacher_strong, mean_teacher_strong = normalize(logits_teacher_strong) if self.logit_stand else logits_teacher_strong
   
        if self.update_teacher:
            logits_attn_weak, ema_alpha_weak = adjust_ema_alpha(self.cfg, epoch, logits_student_weak_may_stand, logits_teacher_weak_may_stand, self.attn)
            logits_attn_strong, ema_alpha_strong = adjust_ema_alpha(self.cfg, epoch, logits_student_strong_may_stand, logits_teacher_strong_may_stand, self.attn)
            with torch.no_grad():
                if self.logit_stand:
                    logits_student_weak_may_shift = denormalize(logits_student_weak_may_stand, std_teacher_weak, mean_teacher_weak) 
                    logits_student_strong_may_shift = denormalize(logits_student_strong_may_stand, std_teacher_strong, mean_teacher_strong) 
                else:
                    logits_student_weak_may_shift = logits_student_weak_may_stand
                    logits_student_strong_may_shift = logits_student_strong_may_stand
                self.teacher.update(index, epoch, 
                                    [logits_student_weak_may_shift, logits_student_strong_may_shift], {}, target, 
                                    ema_alpha=[ema_alpha_weak, ema_alpha_strong])
        else:
            logits_attn_weak, logits_attn_strong = None, None


        batch_size, class_num = logits_student_strong.shape

        pred_teacher_weak = F.softmax(logits_teacher_weak.detach(), dim=1)
        confidence, pseudo_labels = pred_teacher_weak.max(dim=1)
        confidence = confidence.detach()
        conf_thresh = np.percentile(
            confidence.cpu().numpy().flatten(), 50
        )
        mask = confidence.le(conf_thresh).bool()

        class_confidence = torch.sum(pred_teacher_weak, dim=0)
        class_confidence = class_confidence.detach()
        class_confidence_thresh = np.percentile(
            class_confidence.cpu().numpy().flatten(), 50
        )
        class_conf_mask = class_confidence.le(class_confidence_thresh).bool()

        # losses
        loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target))
        loss_kd_weak = self.kd_loss_weight * ((kd_loss(
            logits_student_weak_may_stand,
            logits_teacher_weak_may_stand,
            self.temperature,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak_may_stand,
            logits_teacher_weak_may_stand,
            3.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak_may_stand,
            logits_teacher_weak_may_stand,
            5.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak_may_stand,
            logits_teacher_weak_may_stand,
            2.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak_may_stand,
            logits_teacher_weak_may_stand,
            6.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean())

        loss_kd_strong = self.kd_loss_weight * kd_loss(
            logits_student_strong_may_stand,
            logits_teacher_strong_may_stand,
            self.temperature,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_strong_may_stand,
            logits_teacher_strong_may_stand,
            3.0,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_strong_may_stand,
            logits_teacher_strong_may_stand,
            5.0,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_weak_may_stand,
            logits_teacher_weak_may_stand,
            2.0,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * kd_loss(
            logits_student_weak_may_stand,
            logits_teacher_weak_may_stand,
            6.0,
            logit_stand=self.logit_stand,
        )

        loss_cc_weak = self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
            # reduce=False
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            3.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            5.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        ) * class_conf_mask).mean())
        # loss_cc_strong = self.kd_loss_weight * cc_loss(
        #     logits_student_strong,
        #     logits_teacher_strong,
        #     self.temperature,
        # ) + self.kd_loss_weight * cc_loss(
        #     logits_student_strong,
        #     logits_teacher_strong,
        #     3.0,
        # ) + self.kd_loss_weight * cc_loss(
        #     logits_student_strong,
        #     logits_teacher_strong,
        #     5.0,
        # ) + self.kd_loss_weight * cc_loss(
        #     logits_student_weak,
        #     logits_teacher_weak,
        #     2.0,
        # ) + self.kd_loss_weight * cc_loss(
        #     logits_student_weak,
        #     logits_teacher_weak,
        #     6.0,
        # )
        loss_bc_weak = self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            3.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak, 
            5.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        ) * mask).mean())
        # loss_bc_strong = self.kd_loss_weight * ((bc_loss(
        #     logits_student_strong,
        #     logits_teacher_strong,
        #     self.temperature,
        # ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
        #     logits_student_strong,
        #     logits_teacher_strong,
        #     3.0,
        # ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
        #     logits_student_strong,
        #     logits_teacher_strong,
        #     5.0,
        # ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
        #     logits_student_strong,
        #     logits_teacher_strong,
        #     2.0,
        # ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
        #     logits_student_strong,
        #     logits_teacher_strong,
        #     6.0,
        # ) * mask).mean())
        if logits_attn_weak is not None:
            if epoch >= self.cfg.LEMMA.WARMUP:
                if self.cfg.LEMMA.ATTN.LOSS_DECAY == "exp":
                    self.attn_loss_weight = min(1, np.exp(- self.cfg.LEMMA.ATTN.LOSS_DECAY_RATIO * (epoch - self.cfg.LEMMA.WARMUP))) * self.attn_loss_weight
                elif self.cfg.LEMMA.ATTN.LOSS_DECAY == "jump":
                    self.attn_loss_weight = self.cfg.LEMMA.ATTN.LOSS_WEIGHT_JUMP
            loss_attn = self.attn_loss_weight * (F.cross_entropy(logits_attn_weak, target) + F.cross_entropy(logits_attn_strong, target))
            losses_dict = {
                "loss_ce": loss_ce,
                "loss_kd": loss_kd_weak + loss_kd_strong,
                "loss_cc": loss_cc_weak,
                "loss_bc": loss_bc_weak, 
                "loss_attn": loss_attn,
            }
        else:
            losses_dict = {
                "loss_ce": loss_ce,
                "loss_kd": loss_kd_weak + loss_kd_strong,
                "loss_cc": loss_cc_weak,
                "loss_bc": loss_bc_weak, 
            }
        return logits_student_weak, losses_dict

