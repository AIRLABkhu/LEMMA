import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize(logit):
    mean = logit.mean(dim=-1, keepdim=True)
    stdv = logit.std(dim=-1, keepdim=True)
    return (logit - mean) / (1e-7 + stdv), stdv, mean

def denormalize(logits, std, mean):
    return (logits * std) + mean

def adjust_ema_alpha(cfg, epoch, logits_student, logits_teacher, net=None):
    match cfg.LEMMA.STRATEGY:
        case 'const':
            return None, cfg.LEMMA.EMA_RANGE[0]
        case 'lin':
            ema = (epoch - cfg.LEMMA.WARMUP) / (cfg.SOLVER.EPOCHS - cfg.LEMMA.WARMUP)
            return ema * (cfg.LEMMA.EMA_RANGE[0] - cfg.LEMMA.EMA_RANGE[1]) + cfg.LEMMA.EMA_RANGE[1]
        case 'cos':
            sim = nn.functional.cosine_similarity(logits_student, logits_teacher).unsqueeze(-1)
            _sim = 1 - sim
            return None, (cfg.LEMMA.EMA_RANGE[0] * sim) + (cfg.LEMMA.EMA_RANGE[1] * _sim)
        case 'negcos':
            sim = 1 - nn.functional.cosine_similarity(logits_student, logits_teacher).unsqueeze(-1)
            _sim = 1 - sim
            return None, (cfg.LEMMA.EMA_RANGE[0] * sim) + (cfg.LEMMA.EMA_RANGE[1] * _sim)
        case 'attn':
            if cfg.LEMMA.ISOLATE:
                attn_logit, ema = net(logits_teacher, logits_student.detach(), get_attention=True)
            else:
                attn_logit, ema = net(logits_teacher, logits_student, get_attention=True)
            return attn_logit, ema
        case 'rand':
            batch_size = logits_student.size(0)
            sim = torch.rand(batch_size, 1, device=logits_student.device)
            _sim = 1 - sim
            return None, (cfg.LEMMA.EMA_RANGE[0] * sim) + (cfg.LEMMA.EMA_RANGE[1] * _sim)
        case _:
            raise ValueError


class ConvReg(nn.Module):
    """Convolutional regression"""

    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        else:
            raise NotImplemented("student size {}, teacher size {}".format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)
        
class SimpleAttentionBlock(nn.Module):
    def __init__(self, cfg, num_class):
        super(SimpleAttentionBlock, self).__init__()
        self.attn_dim = cfg.LEMMA.ATTN.DIM
        self.use_cos = cfg.LEMMA.ATTN.COS
        self.invert_attn = cfg.LEMMA.ATTN.INV
        self.num_class = num_class

        if cfg.LEMMA.ATTN.V_LAYER:
            self.q = nn.Linear(self.num_class , self.attn_dim, bias=False)
            self.k = nn.Linear(self.num_class , self.attn_dim, bias=False)
            self.v_s = nn.Linear(self.num_class , self.attn_dim, bias=False)
            self.v_t = nn.Linear(self.num_class , self.attn_dim, bias=False)
            self.fc_out = nn.Linear(self.attn_dim , self.num_class)
        else:
            self.q = nn.Linear(self.num_class , self.attn_dim, bias=False)
            self.k = nn.Linear(self.num_class , self.attn_dim, bias=False)
            self.v_s = nn.Identity()
            self.v_t = nn.Identity()
            if cfg.LEMMA.ATTN.FC_LAYER:
                self.fc_out = nn.Linear(self.num_class , self.num_class)
            else:
                self.fc_out = nn.Identity()

    def forward(self, teacher, student, get_attention=False):
        q = self.q(teacher)
        k = self.k(student)
        v_t = self.v_t(teacher)
        v_s = self.v_s(student)
        
        qk = (q * k).sum(dim=1)
        if self.use_cos:
            attn = qk / (torch.norm(q, dim=1) * torch.norm(k, dim=1))
            attn_score = (attn + 1) / 2.0
        else:
            attn = qk / (self.attn_dim ** (1 / 2))
            attn_score = torch.sigmoid(attn)
        attn_score_ = 1 - attn_score

        if self.invert_attn:
            attn_score_, attn_score = attn_score, attn_score_

        attn_vs = attn_score_.unsqueeze(-1) * v_s
        attn_vt = attn_score.unsqueeze(-1) * v_t

        out = attn_vs + attn_vt
        out = self.fc_out(out)

        if get_attention:
            return out, attn_score
        else:
            return out


def get_feat_shapes(student, teacher, input_size):
    data = torch.randn(1, 3, *input_size)
    with torch.no_grad():
        _, feat_s = student(data)
        _, feat_t = teacher(data)
    feat_s_shapes = [f.shape for f in feat_s["feats"]]
    feat_t_shapes = [f.shape for f in feat_t["feats"]]
    return feat_s_shapes, feat_t_shapes
