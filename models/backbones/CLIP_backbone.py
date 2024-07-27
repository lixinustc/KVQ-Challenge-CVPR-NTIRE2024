from turtle import forward
from numpy import float16
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import sys
from collections import OrderedDict
from collections import OrderedDict
from timm.models.layers import DropPath, trunc_normal_
import sys
sys.path.append('.')
sys.path.append('..')
from .clip import clip
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#import open_clip
'''
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}
'''
from torch.nn import functional as F
import torch
def resize_pos_embed2d(
    posemb,
    src_shape,
    tgt_shape,
    num_prefix_tokens=1,
    interpolation="bicubic",
    antialias=False,
):
    """interpolate positional embedding from src_shape to tgt_shape. posemb: [N,L,C]"""
    if src_shape == tgt_shape:
        return posemb
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = (
            posemb[:, :num_prefix_tokens],
            posemb[:, num_prefix_tokens:],
        )
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb

    posemb_grid = posemb_grid.permute(0, 2, 1).reshape(
        1, -1, src_shape[0], src_shape[1]
    )

    posemb_grid = F.interpolate(
        posemb_grid,
        size=tgt_shape,
        mode=interpolation,
        #antialias=antialias,
        align_corners=False,
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(
        1, tgt_shape[0] * tgt_shape[1], -1
    )
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb



def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    #model_path ='/data0/luyt/Kuaishou_baseline/model/clip/ViT-B-16.pt'
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model

class CLIP(nn.Module):
    def __init__(self, backbone_name, fix_encoder=False,layer_num=1):
        super().__init__()
        self.fix_encoder = fix_encoder
        pretrained_clip_model = load_clip_to_cpu(backbone_name)
        self.img_encoder = pretrained_clip_model.visual
      

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        if self.fix_encoder:
            with torch.no_grad():
                x = self.img_encoder(x)
        else:
            x = self.img_encoder(x)
       
        return x



    
class CLIP_extractor_addadapter_cls(nn.Module):
    def __init__(self, visual,CLIP_location,cls_use):
        super().__init__()
        #assert VPT_type in ['deep', 'shallow']
        self.visual = visual
        self.embed_dim  = self.visual.transformer.width
       
        self.prompt_token_num=1
       
        self.cls_use=cls_use
        
        self.CLIP_location=CLIP_location
        if self.cls_use==True:
            self.adapter_layer=nn.ModuleList()
            for i in range(11-CLIP_location+1):
                self.adapter_layer.append(
                     nn.Sequential(
                                       nn.Linear(self.embed_dim, self.embed_dim // 4),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(self.embed_dim // 4, self.embed_dim),
                                       nn.ReLU(inplace=True),
                                    ) )
        #self.adapter
        self.grid_size = self.visual.grid_size
        
        
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

      
        if self.cls_use==True:
            for param in self.adapter_layer.parameters():
                  param.requires_grad = True
        # Double check
        enabled = set()
        for name, param in self.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        return enabled

    def forward(self, x ):
       
        bb,c,H,W=x.shape
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        _,_,h,w=x.shape

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        pos_embed = self.visual.positional_embedding.to(x.dtype)
        pos_embed = resize_pos_embed2d(pos_embed[None, ...], (self.grid_size,self.grid_size), (h, w))[0]
        
        x = x + pos_embed # shape = [*, grid ** 2 + 1, width]
        x = self.visual.ln_pre(x)  # batch x tokens x dim

        x = x.permute(1, 0, 2)  # NLD -> LND (tokens, batch, dim)
        
        pat_token_list=[]
        
        for i in range(len(self.visual.transformer.resblocks)):
            
            if i<self.CLIP_location:
                x = self.visual.transformer.resblocks[i](x)
                
            else:
                x = self.visual.transformer.resblocks[i](x)
                
                if self.cls_use==True:
                
                    cls_token1 = self.adapter_layer[i-self.CLIP_location](x[:1,])

                    ratio = 0.5 
                    cls_token = ratio * cls_token1 + (1 - ratio) * x[:1,]

                    x = torch.cat((cls_token,x[1:,]),dim=0)
            
        #x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        cls_token = x[:, :1, :]
        pat_token = x[:, 1:, :]
        #qls_token = x[:, num_tokens - self.prompt_token_num:, :]
        N = pat_token.size(1)
        
        cls_attn = torch.cosine_similarity(cls_token,pat_token,dim=-1)#.softmax(dim=-1)
        
        return cls_attn,cls_token,pat_token.unsqueeze(0)

def build_CLIPmodel_basedadapter_cls(backbone_name='ViT-B/16',CLIP_location=None,cls_use=None):
  
    basic_model = load_clip_to_cpu(backbone_name=backbone_name).visual.float()
    
    model = CLIP_extractor_addadapter_cls(visual=basic_model,CLIP_location=CLIP_location,cls_use=cls_use)
    #model = CLIP_extractor_clustertoken(visual=basic_model)
    param=model.freeze()
    #print(param)
    #print('finetune prompt only!!')
    
    return model

