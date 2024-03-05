import time
from functools import partial, reduce

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

import cv2
import random
import os.path as osp
import argparse
from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np
from time import time
from tqdm import tqdm
import pickle
import math
import yaml
from collections import OrderedDict

from functools import reduce
from thop import profile
import copy
import os 


from models.model import VQA_Network 

import sys
sys.path.append('.')
sys.path.append('..')
#sys.path.append('...')
import datasets 



class Trainer:
    def __init__(
        self,
        args,
        config,
        
        
    ):
        super().__init__()
        self.args = args
        self.config=config
        self.gpu_list=[int(item) for item in self.args.gpu_id.split(',')]
        self.device = torch.device("cuda:"+self.args.gpu_id.split(',')[0])
        self.build_datasets()
        self.build_models()
        self.best_results=-1,-1,-1,1999
        self.best_results_ema = -1,-1,-1,1999
        self.key_list = self.config['model']['type'].split(',')

        
    def build_models(self,):
        self.model = VQA_Network(self.config).to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_list)
        if self.config["load_path"] is not None:
            state_dict = torch.load(self.config["load_path"], map_location=self.device)
            if 'state_dict' in state_dict:
                state_dict= state_dict['state_dict']
            else:
                state_dict= state_dict
            #new_state_dict = OrderedDict()
            #for k, v in state_dict.items():
            # ##   name = 'module.'+ k # add `module.`
            #    new_state_dict[name] = v
            
            msg=self.model.load_state_dict(state_dict, strict=False)
            print(msg)


        if self.config["ema"]:
            from copy import deepcopy

            self.model_ema = deepcopy(self.model)
        else:
            self.model_ema = None
            
    def build_optimizer(self,):
        param_groups=[]
        for key, value in dict(self.model.named_children()).items():
            if "backbone" in key:
                param_groups += [
                    {
                        "params": value.parameters(),
                        "lr": self.config["optimizer"]["lr"]
                        * self.config["optimizer"]["backbone_lr_mult"],
                    }
                ]
            else:
                param_groups += [
                    {"params": value.parameters(), "lr": self.config["optimizer"]["lr"]}
                ]

        self.optimizer = torch.optim.AdamW(
            lr=self.config["optimizer"]["lr"],
            params=param_groups,
            weight_decay=self.config["optimizer"]["wd"],
        )

        warmup_iter = 0
        warmup_iter += int(self.config["warmup_epochs"] * len(self.train_loader))
        max_iter = int((self.config["num_epochs"] + self.config["l_num_epochs"]) * len(self.train_loader))
        lr_lambda = (
            lambda cur_iter: cur_iter / warmup_iter
            if cur_iter <= warmup_iter
            else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda,)

    
    def build_datasets(self):

        if 'val' in self.config["data"]:
            val_dataset = getattr(datasets, self.config["data"]["val"]["type"])(self.config["data"]["val"]["args"],None)
            self.val_loader = torch.utils.data.DataLoader( val_dataset, batch_size=1, num_workers=self.config["num_workers"], pin_memory=True,)
        if 'train' in self.config["data"]:
            train_dataset = getattr(datasets, self.config["data"]["train"]["type"])(self.config["data"]["train"]["args"],None)
            self.train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=self.config["batch_size"], num_workers=self.config["num_workers"], shuffle=True,
                )
    

    def train_eval_all_epoches(self,epoch):
        self.model.train()
        for i, data in enumerate(tqdm(self.train_loader, desc=f"Training in epoch {epoch}")):
            self.optimizer.zero_grad()
            
            for key in self.key_list:
                if key in data:
                   data[key] = data[key].to(self.device)
            y = data["label"].float().detach().to(self.device).unsqueeze(-1)
            y_pred = self.model(inputs=data, reduce_scores=False)
            
            loss=0
            for y_pred_idx in range(len(y_pred)):
                p_loss = self.plcc_loss(y_pred[y_pred_idx], y)
                r_loss = self.rank_loss(y_pred[y_pred_idx], y)
                loss += p_loss +0.3 * r_loss
                print(
                        "train",list(data.keys())[y_pred_idx],
                        "train/plcc_loss", p_loss.item(),
                        "train/srcc_loss", r_loss.item(),     
                )
            print("train/total_loss",loss.item())

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if self.model_ema is not None:
                model_params = dict(self.model.named_parameters())
                model_ema_params = dict(self.model_ema.named_parameters())
                for k in model_params.keys():
                    model_ema_params[k].data.mul_(0.999).add_(
                        model_params[k].data, alpha=1 - 0.999
                )
        self.model.eval()

        self.best_results = self.inferece_per_epoch(self.model,self.best_results,suffix='n')
        self.best_results_ema = self.inferece_per_epoch(self.model_ema,self.best_results_ema,suffix='s')

        return self.best_results,self.best_results_ema

    def inferece_per_epoch(self,model,best,suffix):

        best_s, best_p, best_k, best_r = best
        results = []
        
        for i, data in enumerate(tqdm(self.val_loader, desc="Validating")):
            result={}
            
            for key in self.key_list:
                if key in data:
                    data[key] = data[key].to(self.device)
                    b, c, t, h, w = data[key].shape
                    data[key] = (
                        data[key]
                        .reshape(
                            b, c, data["num_clips"][key], t // data["num_clips"][key], h, w
                        )
                        .permute(0, 2, 1, 3, 4, 5)
                        .reshape(
                            b * data["num_clips"][key], c, t // data["num_clips"][key], h, w
                        )
                    )
            with torch.no_grad():
                result["pred"] = model(inputs=data,reduce_scores=True).cpu().numpy()
            result["label"] = data["label"].item()
            del data
            results.append(result)

        labels = [r["label"] for r in results]
        preds = [np.mean(r["pred"][:]) for r in results]
        preds = self.rescale(preds, labels)

        s = spearmanr(labels, preds)[0]
        p = pearsonr(labels, preds)[0]
        k = kendallr(labels, preds)[0]
        r = np.sqrt(((labels - preds) ** 2).mean())

        print('SRCC{}PLCC{}KRCC{}RMSE{}'.format(s,p,k,r))
        if s + p > best_s + best_p :
            state_dict = model.state_dict()

            save_name=self.config["name"] + "_head_" + self.args.test_set 
            torch.save(
                    {"state_dict": state_dict, "validation_results": best,},
                    f"{self.args.resume}/{save_name}_{suffix}_finetuned.pth",
                )
        
        best_s, best_p, best_k, best_r = (
            max(best_s, s),
            max(best_p, p),
            max(best_k, k),
            min(best_r, r),
        )
        print(
            {
                f"val_{suffix}/best_SRCC-{suffix}": best_s,
                f"val_{suffix}/best_PLCC-{suffix}": best_p,
                f"val_{suffix}/best_KRCC-{suffix}": best_k,
                f"val_{suffix}/best_RMSE-{suffix}": best_r,
            }
        )

        return best_s, best_p, best_k, best_r
    
    

    def inferece(self):

        output_results=[]
        

        for i, data in enumerate(tqdm(self.val_loader, desc="Validating")):
            result={}
            self.model.eval()
            for key in self.key_list:
                if key in data:
                    data[key] = data[key].to(self.device)
                    b, c, t, h, w = data[key].shape
                    data[key] = (
                        data[key]
                        .reshape(
                            b, c, data["num_clips"][key], t // data["num_clips"][key], h, w
                        )
                        .permute(0, 2, 1, 3, 4, 5)
                        .reshape(
                            b * data["num_clips"][key], c, t // data["num_clips"][key], h, w
                        )
                    )
            with torch.no_grad():
                pred = self.model(inputs=data,reduce_scores=True).cpu().numpy()
            
            output_results.append((data["video_name"][0],pred.mean(0).item()))
        
        with open('output.txt',"w") as file:
            for item in output_results:
                line =f"{item[0]},{item[1]}\n"
                file.write(line)    
        

    def rank_loss(self,y_pred,y):
        ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
        scale = 1 + torch.max(ranking_loss)
        return (
            torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
        ).float()

    def plcc_loss(self,y_pred, y):
        sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
        y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
        sigma, m = torch.std_mean(y, unbiased=False)
        y = (y - m) / (sigma + 1e-8)
        loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
        rho = torch.mean(y_pred * y)
        loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
        return ((loss0 + loss1) / 2).float()
        
    def rescale(self,pr, gt=None):
        if gt is None:
            pr = (pr - np.mean(pr)) / np.std(pr)
        else:
            pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
        return pr
