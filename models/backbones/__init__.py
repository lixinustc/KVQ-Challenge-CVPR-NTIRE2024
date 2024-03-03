from .conv_backbone import convnext_3d_small, convnext_3d_tiny
#from .head import IQAHead, VARHead, VQAHead, MaxVQAHead,simpleVQAHead
from .swin_backbone import SwinTransformer2D as IQABackbone
from .swin_backbone import SwinTransformer3D as VQABackbone
from .swin_backbone import swin_3d_small, swin_3d_tiny
from .simpleVQA_model import resnet50

__all__ = [
    "VQABackbone",
    "IQABackbone",
    "VQAHead",
    "MaxVQAHead",
    "IQAHead",
    "VARHead",
    "simpleVQAHead",
    "BaseEvaluator",
    "BaseImageEvaluator",
    "DOVER",
    "resnet50"
]
