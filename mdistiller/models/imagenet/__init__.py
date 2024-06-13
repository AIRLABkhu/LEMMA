import os
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .mobilenetv2 import MobileNetV2
from ..memory import H5Memory as Memory, AugMemory


imagenet_model_prefix = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "../../../download_ckpts/imagenet_teachers/"
)
imagenet_model_dict = {
    "ResNet18": resnet18,
    "ResNet34": resnet34,
    "ResNet50": resnet50,
    "ResNet101": resnet101,
    "MobileNetV2": MobileNetV2,
    
    "ResNet18_mem": (Memory, imagenet_model_prefix + "ResNet18_memory"),
    "ResNet34_mem": (Memory, imagenet_model_prefix + "ResNet34_memory"),
    "ResNet50_mem": (Memory, imagenet_model_prefix + "ResNet50_memory"),
    "ResNet101_mem": (Memory, imagenet_model_prefix + "ResNet101_memory"),
    "MobileNetV2_mem": (Memory, imagenet_model_prefix + "MobileNetV2_memory"),
}
