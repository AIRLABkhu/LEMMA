import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import SimpleAttentionBlock


class Distiller(nn.Module):
    def __init__(self, student, teacher, cfg=None):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
        if cfg is not None:
            self.update_teacher = all([
                cfg.LEMMA.ENABLE, 
                cfg.LEMMA.EMA_RANGE is not None, 
                not (cfg.LEMMA.EMA_RANGE[0] == cfg.LEMMA.EMA_RANGE[1] == 1), 
            ])
        else:
            self.update_teacher = False
        
        if cfg.LEMMA.STRATEGY == 'attn':  
            num_classes = {'cifar100': 100, 'imagenet': 1000}[cfg.DATASET.TYPE]
            self.attn = SimpleAttentionBlock(cfg, num_classes)
        else:
            self.attn = None

    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [
            *[v for k, v in self.student.named_parameters()],
            *(self.attn.parameters() if self.attn else []),
        ]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0

    def forward_train(self, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()

    def forward_test(self, image):
        return self.student(image)[0]

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])


class Vanilla(nn.Module):
    def __init__(self, student):
        super(Vanilla, self).__init__()
        self.student = student

    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()]

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        loss = F.cross_entropy(logits_student, target)
        return logits_student, {"ce": loss}

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])

    def forward_test(self, image):
        return self.student(image)[0]
