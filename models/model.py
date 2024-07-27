        
import time
from functools import partial, reduce

import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool3d
from models.backbones.swin_backbone import swin_3d_small, swin_3d_tiny
from models.backbones.conv_backbone import convnext_3d_small, convnext_3d_tiny, convnextv2_3d_pico, convnextv2_3d_femto
from .head import IQAHead, VARHead, VQAHead ,simpleVQAHead
from models.backbones.swin_backbone import SwinTransformer2D as ImageBackbone
from models.backbones.swin_backbone import SwinTransformer3D as VideoBackbone
from models.backbones.simpleVQA_model import resnet50 as simpleVQA_Backbone
from models.backbones.KSVQE_model import KSVQE as KSVQE_Backbone
import torchvision.transforms as T
import numpy as np 
        
class VQA_Network(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()        
        self.config=config
        self.key_names=[]
        self.multi=False
        self.layer=-1
        for key, hypers in config['model']['args'].items():
            
            if key == "swin_tiny":
                backbone = swin_3d_tiny(**hypers['backbone'])
                head = VQAHead(**hypers['head'])
                self.key_names.append(key)
            elif key == "swin_tiny_grpb":
                # to reproduce fast-vqa
                backbone = VideoBackbone()
                head = VQAHead(**hypers['head'])
                self.key_names.append(key)
            elif key == "swin_tiny_grpb_m":
                # to reproduce fast-vqa-m
                backbone = VideoBackbone(window_size=(4, 4, 4), frag_biases=[0, 0, 0, 0])
                self.key_names.append(key)
                head = VQAHead(**hypers['head'])
            elif key == "swin_small":
                backbone = swin_3d_small(**hypers['backbone'])
                head = VQAHead(**hypers['head'])
                self.key_names.append(key)
            elif key == "conv_tiny":
                backbone = convnext_3d_tiny(pretrained=True)
                head = VQAHead(**hypers['head'])
                self.key_names.append(key)
            elif key == "simpleVQA":
                backbone = simpleVQA_Backbone(pretrained=True)
                head = simpleVQAHead(**hypers['head'])
                self.key_names.append(key)
            elif key == "KSVQE":
                                     
                backbone =  KSVQE_Backbone(num_samples=hypers['backbone']['num_samples'],
                                            sample_type=hypers['backbone']['sample_type'],
                
                                            CLIP_location=hypers['backbone']['CLIP_location'],
                                            cls_use=hypers['backbone']['cls_use'],
                                            tuning_stage = hypers['backbone']['tuning_stage'],
                                            a1 = hypers['backbone']['a1'],
                                            a2 = hypers['backbone']['a2'],
                                            frozen_stages=hypers['backbone']['frozen_stages'],
                                            )
                head = VQAHead(**hypers['head'])
                self.key_names.append(key)
            else:
                raise NotImplementedError

            print("Setting backbone:", key + "_backbone")
            setattr(self, key + "_backbone", backbone)
            print("Setting head:", key + "_head")
            setattr(self, key + "_head", head)
    def forward(
        self,
        inputs,
        targets=None,
        inference=True,
        return_pooled_feats=False,
        reduce_scores=False,
        pooled=False,
        clip_return=False,
        **kwargs
    ):
        
     
        scores = []
        feats = {}
        dis_contra_loss = None
        for key in self.key_names:
            if key == 'KSVQE':
                feat, dis_contra_loss = getattr(self, key + "_backbone")(
                    inputs, multi=self.multi, layer=self.layer, **kwargs
                )
            else:
                feat = getattr(self, key + "_backbone")(
                    inputs, multi=self.multi, layer=self.layer, **kwargs
                )
            scores += [getattr(self, key + "_head")(feat)]
            if return_pooled_feats:
                feats[key] = feat
        if reduce_scores:
            if len(scores) > 1:
                scores = reduce(lambda x, y: x + y, scores)
            else:
                scores = scores[0]
        
        
        if return_pooled_feats:
            if dis_contra_loss is  not None:
                 return scores, feats, dis_contra_loss
            else:
                 return scores, feats
            
        if dis_contra_loss is not None:
            return scores, dis_contra_loss
        else:
            return scores
