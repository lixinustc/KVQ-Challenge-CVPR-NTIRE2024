import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from einops import rearrange
from math import sqrt

class PredictorLG(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.GELU(),
           
        )

    def forward(self, x):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:,:, :C//2]
        
        global_x = torch.mean(x[:,:, C//2:], dim=1, keepdim=True)
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)
        return self.out_conv(x)
    
class PredictorLG_conv(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            #nn.LayerNorm(embed_dim),
            #nn.Linear(embed_dim, embed_dim),
            #nn.GELU()
            
            nn.Conv2d(embed_dim, 2, kernel_size=3, stride=1,padding=1),
            nn.GELU(),
            nn.Conv2d(2,2, kernel_size=3, stride=1,padding=1),
            nn.GELU(),
            nn.Softmax(),
            
        )

       
    def forward(self, x):
        x = self.in_conv(x)
        
       
        return x

def HardTopK(k, x):
    topk_results = torch.topk(x, k=k, dim=-1, sorted=False)
    indices = topk_results.indices # b, k
    indices = torch.sort(indices, dim=-1).values
    return indices

def GumbelTopK(k, x):
    z= -torch.log(-torch.log(torch.rand_like(x)))
    topk_results = torch.topk(x+z, k=k, dim=-1, sorted=False)
    indices = topk_results.indices # b, k
    indices = torch.sort(indices, dim=-1).values
    return indices

class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 1000):
        super(PerturbedTopK, self).__init__()
        self.num_samples = num_samples
        self.k = k

    def __call__(self, x, sigma):
        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, sigma)


class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 1000, sigma: float = 0.05):
        b, d = x.shape
        # for Gaussian: noise and gradient are the same.
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)
        perturbed_x = x[:, None, :] + noise * sigma # b, nS, d
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices # b, nS, k
        indices = torch.sort(indices, dim=-1).values # b, nS, k

        # b, nS, k, d
        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
        indicators = perturbed_output.mean(dim=1) # b, k, d

        # constants for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        # tensors for backward
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise
        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise

        # import pdb; pdb.Pdb(nosigint=True).set_trace()
        if ctx.sigma <= 1e-20:
            b, _, k, d = ctx.perturbed_output.size()
            expected_gradient = torch.zeros(b, k, d).to(grad_output.device)
        else:
            expected_gradient = (
                torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
                / ctx.num_samples
                / (ctx.sigma)
            )

        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)

        return (grad_input,) + tuple([None] * 5)


def batched_index_select(input, dim, index):
    for i in range(1, len(input.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def extract_patches_from_indices(x, indices):
    batch_size, _, channels = x.shape
    k = indices.shape[-1]
    patches = x
    patches = batched_index_select(patches, 1, indices)
    patches = patches.contiguous().view(batch_size, k, channels)
    return patches


def extract_patches_from_indicators(x, indicators):
    indicators = rearrange(indicators, "b d k -> b k d") #64,49,196  
    patches = torch.bmm(indicators, x)# 64 49 196  64 196 Chw  64 49 Chw
    return patches



def min_max_norm(x):
    flatten_score_min = x.min(axis=-1, keepdim=True).values
    flatten_score_max = x.max(axis=-1, keepdim=True).values
    norm_flatten_score = (x - flatten_score_min) / (flatten_score_max - flatten_score_min + 1e-5)
    return norm_flatten_score



class PatchNet_ms(nn.Module):
    def __init__(self, score, k, in_channels, stride=None, num_samples=500):
        super(PatchNet_ms, self).__init__()
        self.k = k
        self.anchor_size = int(sqrt(k))
      
        self.stride = stride
        self.score = score
        self.in_channels = in_channels
        self.num_samples = num_samples

        if score == 'tpool':
            self.score_network = PredictorLG(embed_dim=2*in_channels)

        elif score == 'spatch':
            self.score_network = PredictorLG(embed_dim=in_channels)
          
    
    def get_indicator(self, scores, k, sigma):
        indicator = PerturbedTopKFunction.apply(scores, k, self.num_samples, sigma)
        indicator = einops.rearrange(indicator, "b k d -> b d k")
        return indicator
    
    def get_indices(self, scores, k):
        indices = HardTopK(k, scores)
        return indices
    
    def generate_random_indices(self, b, n, k):
        indices = []
        for _ in range(b):
            indice = np.sort(np.random.choice(n, k, replace=False))
            indices.append(indice)
        indices = np.vstack(indices)
        indices = torch.Tensor(indices).long().cuda()
        return indices
    
    def generate_uniform_indices(self, b, n, k):
        indices = torch.linspace(0, n-1, steps=k).long()
        indices = indices.unsqueeze(0).cuda()
        indices = indices.repeat(b, 1)
        return indices


    def forward(self, x, type, H,W, T, sigma):
        B = x.size(0)
        
        indicator = None
        indices = None

        if type == 'time':
            if self.score == 'tpool':
                x = rearrange(x, 'b c t h w -> b t (h w) c')
                avg = torch.mean(x, dim=2, keepdim=False)
                max_ = torch.max(x, dim=2).values
                x_ = torch.cat((avg, max_), dim=2)
                scores = self.score_network(x_).squeeze(-1)
                scores = min_max_norm(scores)
                
                if self.training:
                    indicator = self.get_indicator(scores, self.k, sigma)
                else:
                    indices = self.get_indices(scores, self.k)
                x = rearrange(x, 'b t n c -> b t (n c)')
            
        else:
            s = self.stride if self.stride is not None else int(max((W - self.anchor_size) // 2, 1))
                
            if self.score == 'spatch':
                x = rearrange(x, 'b t c h w -> (b t) (h w) c')
                #print(x.shape)
                scores = self.score_network(x)
                scores = rearrange(scores, '(b t) (h w) c -> (b t) c h w', b=B, h=H)
                scores = F.unfold(scores, kernel_size=self.anchor_size, stride=s)
                scores = scores.mean(dim=1)
                scores = min_max_norm(scores)
                
                x = rearrange(x, '(b t) (h w) c -> (b t) c h w', b=B, h=H)
                x = F.unfold(x, kernel_size=self.anchor_size, stride=s).permute(0, 2, 1).contiguous()

                if self.training:
                    indicator = self.get_indicator(scores, 1, sigma)
                    
                else:
                    indices = self.get_indices(scores, 1)


        if self.training:
            if indicator is not None:
                patches = extract_patches_from_indicators(x, indicator)

            elif indices is not None:
                patches = extract_patches_from_indices(x, indices)
                
            if type == 'time':
                patches = rearrange(patches, 'b k (h w c) -> b c k h w', h=H, w=W)

            elif self.score == 'spatch':
                patches = patches.squeeze(1)
                patches = rearrange(patches, '(b t) (c kh kw) -> b c t kh kw', b=B, c=self.in_channels, kh=self.anchor_size)
            

            return patches
            
        
        else:
            patches = extract_patches_from_indices(x, indices)
            if type == 'time':
                patches = rearrange(patches, 'b k (h w c) -> b c k h w', h=H, w=W)

            elif self.score == 'spatch':
                patches = patches.squeeze(1)
                patches = rearrange(patches, '(b t) (c kh kw) -> b c t kh kw', b=B, c=self.in_channels, kh=self.anchor_size)

            return patches



class PatchNet_ms_conv(nn.Module):
    def __init__(self, score, k, in_channels, stride=None, num_samples=500):
        super(PatchNet_ms_conv, self).__init__()
        self.k = k
        self.anchor_size = int(sqrt(k))
      
        self.stride = stride
        self.score = score
        self.in_channels = in_channels
        self.num_samples = num_samples

        if score == 'tpool':
            self.score_network = PredictorLG_conv(embed_dim=2*in_channels)

        elif score == 'spatch':
            self.score_network = PredictorLG_conv(embed_dim=2*in_channels)
          
    
    def get_indicator(self, scores, k, sigma):
        indicator = PerturbedTopKFunction.apply(scores, k, self.num_samples, sigma)
        indicator = einops.rearrange(indicator, "b k d -> b d k")
        return indicator
    
    def get_indices(self, scores, k):
        indices = HardTopK(k, scores)
        return indices
    
    def generate_random_indices(self, b, n, k):
        indices = []
        for _ in range(b):
            indice = np.sort(np.random.choice(n, k, replace=False))
            indices.append(indice)
        indices = np.vstack(indices)
        indices = torch.Tensor(indices).long().cuda()
        return indices
    
    def generate_uniform_indices(self, b, n, k):
        indices = torch.linspace(0, n-1, steps=k).long()
        indices = indices.unsqueeze(0).cuda()
        indices = indices.repeat(b, 1)
        return indices


    def forward(self, x, type):
        B = x.size(0)
        
        indicator = None
        indices = None

        if type == 'time':
            if self.score == 'tpool':
                x = rearrange(x, 'b c t h w -> b t (h w) c')
                avg = torch.mean(x, dim=2, keepdim=False)
                max_ = torch.max(x, dim=2).values
                x_ = torch.cat((avg, max_), dim=2)
                scores = self.score_network(x_).squeeze(-1)
                scores = min_max_norm(scores)
                
                if self.training:
                    indicator = self.get_indicator(scores, self.k, sigma)
                else:
                    indices = self.get_indices(scores, self.k)
                x = rearrange(x, 'b t n c -> b t (n c)')
            
        else:
            
                
            if self.score == 'spatch':
                b,t,c,h,w=x.shape
                x = rearrange(x, 'b t c h w -> (b t) c h w')
                scores = self.score_network(x) # (b t) k h w

                x_scale1,x_scale2=x.chunk(2, dim = -3)
                #scores = rearrange(scores, '(b t) (h w) c -> (b t) c h w', b=B, h=H)
                xa = scores[:,0,...].unsqueeze(1).repeat(1,c//2,1,1)* x_scale1+scores[:,1,...].unsqueeze(1).repeat(1,c//2,1,1)* x_scale2

                xa = rearrange(xa, '(b t) c h w -> b c t h w',t=t)
                
             
        return xa
            


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = True, dim: int = -1) -> torch.Tensor:
    # _gumbels = (-torch.empty_like(
    #     logits,
    #     memory_format=torch.legacy_contiguous_format).exponential_().log()
    #             )  # ~Gumbel(0,1)
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape) #gumbel.shape:([1,16,1])

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret
        
class RegionNet_CLIP(nn.Module):
    def __init__(self,  k,anchor_size,stride, num_samples=500,sample_type='topkpertubation'): #
        super(RegionNet_CLIP, self).__init__()
        self.k = k #49
        self.stride=stride
        self.anchor_size =anchor_size
        self.num_samples=num_samples
        self.sample_type = sample_type
    
    def get_indicator(self, scores, k, sigma):
        indicator = PerturbedTopKFunction.apply(scores, k, self.num_samples, sigma)
        indicator = einops.rearrange(indicator, "b k d -> b d k")
        return indicator
    
    def get_gumbel_indicator(self, scores, k, sigma):
        b, d = scores.shape
        indicator = gumbel_softmax(scores, hard=True) #b,d
        indicator = indicator.unsqueeze(1)
        indicator = einops.rearrange(indicator, "b k d -> b d k")
        return indicator
    
    def get_multinomial_indicator(self, scores, k, sigma):
        b, d = scores.shape
        idex = torch.multinomial(w_fre1, num_samples=self.args.slt_num)
        indicator = torch.nn.functional.one_hot(idex, num_classes=d).float()
        
        indicator = indicator.unsqueeze(1)
        indicator = einops.rearrange(indicator, "b k d -> b d k")
        return indicator
    
    def get_indices(self, scores, k):
        indices = HardTopK(k, scores)
        return indices
    
    def generate_random_indices(self, b, n, k):
        indices = []
        for _ in range(b):
            indice = np.sort(np.random.choice(n, k, replace=False))
            indices.append(indice)
        indices = np.vstack(indices)
        indices = torch.Tensor(indices).long().cuda()
        return indices
    
    def generate_uniform_indices(self, b, n, k):
        indices = torch.linspace(0, n-1, steps=k).long().cuda()
        indices = indices.unsqueeze(0).cuda()
        indices = indices.repeat(b, 1)
        return indices

    def extend_fullcls_indicator(self,score,group_id):
        # cls_attn B,N_key
        # group_id B,T
        # group_mask B,N_key
        B,N_key,L,n = score.shape
        B,T = group_id.shape
        full_score=score.new_zeros((B,T,L,n))
        for i in range(B):
            for j in range(T):
                    full_score[i,j] = score[i,int(group_id[i,j].item())]
        return full_score
    def extend_fullcls_indices(self,score,group_id):
        # cls_attn B,N_key
        # group_id B,T
        # group_mask B,N_key
        B,N_key,n = score.shape
        B,T = group_id.shape
        full_score=score.new_zeros((B,T,n))
        for i in range(B):
            for j in range(T):
                    full_score[i,j] = score[i,int(group_id[i,j].item())]
        return full_score
    def forward(self, x, score, sigma,group_id,extra_score=None): #score: B,N
        B = x.size(0)
        
        indicator = None
        indices = None
    
        b,c,t,h,w = x.shape  
        b,n_key,L=score.shape
        score = score.view(b*n_key,1,int(sqrt(L)),int(sqrt(L)))
        
        #self.anchor_size =32# h// 14 #orgi fragment sample 2 multiply grid
        gh = h//self.anchor_size 
        gw = w//self.anchor_size 
        if score.shape[-1]!=gw or score.shape[-2]!=gh:
            score = torch.nn.functional.interpolate(
                    score, scale_factor=( gh/score.shape[-2], gw/score.shape[-1]), mode="nearest"
                )
        if extra_score is not None:
            extra_score=extra_score.view(b*n_key,1,gh,gw)
            score=score*extra_score
        
        x = rearrange(x, 'b c t h w -> (b t) c h w', b=B, h=h)
        x = (x.contiguous()
            .view(b*t,c,h//self.anchor_size,self.anchor_size,w//self.anchor_size,self.anchor_size)
            .permute(0,1,3,5,2,4) 
            .contiguous()
            .view(b*t, c*self.anchor_size*self.anchor_size,h//self.anchor_size, (w//self.anchor_size))) #B,C,N,kermkerne
        x = F.unfold(x, kernel_size=int(sqrt(self.k)), stride=self.stride).permute(0, 2, 1).contiguous()
        scores = F.unfold(score, kernel_size=int(sqrt(self.k)), stride=self.stride)
        scores = scores.mean(dim=1)
        scores = min_max_norm(scores)
        num_region = scores.size(1)
        if self.training:
            if self.sample_type=='topkpertubation':
                indicator = self.get_indicator(scores,1, sigma) #b*n_key, 1, N_region
                indicator =indicator.contiguous().view(b,n_key,num_region,1)
                full_indicator=self.extend_fullcls_indicator(indicator,group_id)
                full_indicator =full_indicator.contiguous().view(b*t,num_region,1)
            elif self.sample_type=='gumbel':
                indicator = self.get_gumbel_indicator(scores,1, sigma) #b*n_key, 49,L
                indicator =indicator.contiguous().view(b,n_key,num_region,1)
                full_indicator=self.extend_fullcls_indicator(indicator,group_id)
                full_indicator =full_indicator.contiguous().view(b*t,num_region,1)
                
            elif self.sample_type=='multinomial':
                b,d=scores.shape
                indicator = self.get_multinomial_indicator(scores,1, sigma) #b*n_key, 49,L
                indicator =indicator.contiguous().view(b,n_key,num_region,1)
                full_indicator=self.extend_fullcls_indicator(indicator,group_id)
                full_indicator =full_indicator.contiguous().view(b*t,num_region,1)
            elif self.sample_type=='random':
               
                indices = self.generate_random_indices( b*n_key, num_region, 1)#self.get_indices(score_row, self.kh)
                indices=indices.view(b,n_key,1)
                full_indices=self.extend_fullcls_indices(indices,group_id)
                full_indices =full_indices.contiguous().view(b*t,1)
        else:
            if self.sample_type=='random':
               
                indices = self.generate_random_indices( b*n_key, num_region, 1)#self.get_indices(score_row, self.kh)
                indices=indices.view(b,n_key,1)
                full_indices=self.extend_fullcls_indices(indices,group_id)
                full_indices =full_indices.contiguous().view(b*t,1)
            else:
                indices = self.get_indices(scores, 1)
                indices=indices.view(b,n_key,1)
                full_indices=self.extend_fullcls_indices(indices,group_id)
                full_indices =full_indices.contiguous().view(b*t,1)
        if self.training:
            if indicator is not None:
                patches = extract_patches_from_indicators(x, full_indicator)

            elif full_indices is not None:
                patches = extract_patches_from_indices(x, full_indices)
            patches = patches.squeeze(1)
           
            patches = rearrange(patches, '(b t) (chw kh kw) -> b t chw  kh kw', b=B, chw=c*self.anchor_size*self.anchor_size, kh=int(sqrt(self.k)), kw=int(sqrt(self.k)))
            patches = (patches.view(b,t,c,self.anchor_size,self.anchor_size,int(sqrt(self.k)),int(sqrt(self.k))).permute(0,2,1,5,3,6,4)
                      .contiguous()
                     .view(b,c,t,int(sqrt(self.k))*self.anchor_size,int(sqrt(self.k))*self.anchor_size))
            return patches
        else:
            patches = extract_patches_from_indices(x, full_indices)
            patches = patches.squeeze(1)
            n=patches.shape[1] #num_grid
            patches = rearrange(patches, '(b t) (chw kh kw) -> b t chw  kh kw', b=B, chw=c*self.anchor_size*self.anchor_size, kh=int(sqrt(self.k)), kw=int(sqrt(self.k)))
            patches = (patches.view(b,t,c,self.anchor_size,self.anchor_size,int(sqrt(self.k)),int(sqrt(self.k))).permute(0,2,1,5,3,6,4)
                      .contiguous()
                     .view(b,c,t,int(sqrt(self.k))*self.anchor_size,int(sqrt(self.k))*self.anchor_size))
            return patches


