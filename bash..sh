CUDA_VISIBLE_DEVICES=7 python tools/train.py --cfg configs/cifar100_lemma/kd/resnet32x4_resnet8x4.yaml --name kd_teacher/CosineAnnealingLR/ema_stand_00 --logit-stand DISTILLER.EMA_SCHEDULER CosineAnnealingLR
CUDA_VISIBLE_DEVICES=7 python tools/train.py --cfg configs/cifar100_lemma/kd/resnet32x4_resnet8x4.yaml --name kd_teacher/CosineAnnealingLR/ema_stand_01 --logit-stand DISTILLER.EMA_SCHEDULER CosineAnnealingLR


CUDA_VISIBLE_DEVICES=6 python tools/train.py --cfg configs/cifar100_lemma/kd/resnet32x4_resnet8x4.yaml --name kd_teacher/CosineAnnealingLR/ema_stand_02 --logit-stand DISTILLER.EMA_SCHEDULER CosineAnnealingLR
CUDA_VISIBLE_DEVICES=6 python tools/train.py --cfg configs/cifar100_lemma/kd/resnet32x4_resnet8x4.yaml --name kd_teacher/CosineAnnealingLR/ema_stand_03 --logit-stand DISTILLER.EMA_SCHEDULER CosineAnnealingLR
CUDA_VISIBLE_DEVICES=6 python tools/train.py --cfg configs/cifar100_lemma/kd/resnet32x4_resnet8x4.yaml --name kd_teacher/CosineAnnealingLR/ema_stand_04 --logit-stand DISTILLER.EMA_SCHEDULER CosineAnnealingLR

CUDA_VISIBLE_DEVICES=5 python tools/train.py --cfg configs/cifar100_lemma/kd/resnet32x4_resnet8x4.yaml --name kd_teacher/CosineAnnealingWarmRestarts/ema_stand_00 --logit-stand DISTILLER.EMA_SCHEDULER CosineAnnealingWarmRestarts
CUDA_VISIBLE_DEVICES=5 python tools/train.py --cfg configs/cifar100_lemma/kd/resnet32x4_resnet8x4.yaml --name kd_teacher/CosineAnnealingWarmRestarts/ema_stand_01 --logit-stand DISTILLER.EMA_SCHEDULER CosineAnnealingWarmRestarts

CUDA_VISIBLE_DEVICES=4 python tools/train.py --cfg configs/cifar100_lemma/kd/resnet32x4_resnet8x4.yaml --name kd_teacher/CosineAnnealingWarmRestarts/ema_stand_02 --logit-stand DISTILLER.EMA_SCHEDULER CosineAnnealingWarmRestarts
CUDA_VISIBLE_DEVICES=4 python tools/train.py --cfg configs/cifar100_lemma/kd/resnet32x4_resnet8x4.yaml --name kd_teacher/CosineAnnealingWarmRestarts/ema_stand_03 --logit-stand DISTILLER.EMA_SCHEDULER CosineAnnealingWarmRestarts
CUDA_VISIBLE_DEVICES=4 python tools/train.py --cfg configs/cifar100_lemma/kd/resnet32x4_resnet8x4.yaml --name kd_teacher/CosineAnnealingWarmRestarts/ema_stand_04 --logit-stand DISTILLER.EMA_SCHEDULER CosineAnnealingWarmRestarts

CUDA_VISIBLE_DEVICES=3 python tools/train.py --cfg configs/cifar100_lemma/kd/resnet32x4_resnet8x4.yaml --name kd_teacher/MultiStepLR/ema_stand_00 --logit-stand DISTILLER.EMA_SCHEDULER MultiStepLR
CUDA_VISIBLE_DEVICES=3 python tools/train.py --cfg configs/cifar100_lemma/kd/resnet32x4_resnet8x4.yaml --name kd_teacher/MultiStepLR/ema_stand_01 --logit-stand DISTILLER.EMA_SCHEDULER MultiStepLR

CUDA_VISIBLE_DEVICES=2 python tools/train.py --cfg configs/cifar100_lemma/kd/resnet32x4_resnet8x4.yaml --name kd_teacher/MultiStepLR/ema_stand_02 --logit-stand DISTILLER.EMA_SCHEDULER MultiStepLR
CUDA_VISIBLE_DEVICES=2 python tools/train.py --cfg configs/cifar100_lemma/kd/resnet32x4_resnet8x4.yaml --name kd_teacher/MultiStepLR/ema_stand_03 --logit-stand DISTILLER.EMA_SCHEDULER MultiStepLR
CUDA_VISIBLE_DEVICES=2 python tools/train.py --cfg configs/cifar100_lemma/kd/resnet32x4_resnet8x4.yaml --name kd_teacher/MultiStepLR/ema_stand_04 --logit-stand DISTILLER.EMA_SCHEDULER MultiStepLR

CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg configs/cifar100_lemma/kd/resnet32x4_resnet8x4.yaml --name kd_teacher/Linear/ema_stand_00 --logit-stand DISTILLER.EMA_SCHEDULER Linear
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg configs/cifar100_lemma/kd/resnet32x4_resnet8x4.yaml --name kd_teacher/Linear/ema_stand_01 --logit-stand DISTILLER.EMA_SCHEDULER Linear


CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100_lemma/kd/resnet32x4_resnet8x4.yaml --name kd_teacher/Linear/ema_stand_02 --logit-stand DISTILLER.EMA_SCHEDULER Linear
CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100_lemma/kd/resnet32x4_resnet8x4.yaml --name kd_teacher/Linear/ema_stand_03 --logit-stand DISTILLER.EMA_SCHEDULER Linear
CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100_lemma/kd/resnet32x4_resnet8x4.yaml --name kd_teacher/Linear/ema_stand_04 --logit-stand DISTILLER.EMA_SCHEDULER Linear
