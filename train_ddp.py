import torch
torch.autograd.set_detect_anomaly(True)
import argparse
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

from trainer_ddp import Trainer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ['TORCH_DISTRIBUTED_DEBUG ']= 'DETAIL'

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="config/kwai_simpleVQA_real.yml", help="the option file"
    )

    parser.add_argument(
        "-t", "--train_set", type=str, default="train", help="target_set"
    )
    parser.add_argument(
        "-t1", "--test_set", type=str, default="val-ltest", help="target_set"
    )
    parser.add_argument(
        "-r", "--resume", type=str, default="./checkpoint/", help="target_set"
    )

    parser.add_argument('--gpu_id',type=str,default='1,2')


    args = parser.parse_args()
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    print(opt)

    ## adaptively choose the device
    if opt["ddp"] == True:
        rank = int(os.environ["RANK"])
        print(os.environ["LOCAL_RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank % torch.cuda.device_count())
        world_size= torch.cuda.device_count()
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
    else:
        device = torch.device("cuda:"+args.gpu_id.split(',')[0])
    
    if opt['ddp']==False or (opt['ddp'] and local_rank==0):
        print(opt)
    if opt['ddp']==False or (opt['ddp'] and local_rank==0):
        if not os.path.exists(args.resume):
                os.makedirs(args.resume)

    
   
        

    trainer=Trainer(args, opt,)
    
    trainer.build_datasets()
    trainer.build_models(device, local_rank)
    trainer.build_optimizer()
    
    for epoch in range(opt["num_epochs"]):

        if opt['ddp']==False or (opt['ddp'] and local_rank==0):
                print(f"Finetune Epoch {epoch}:")

        bests,bests_n=trainer.train_eval_all_epoches(epoch, device, local_rank ,world_size)
        if opt["num_epochs"] >= 0 and local_rank==0:
            
            print(
                f"""
                the best validation accuracy of the model-s is as follows:
                SROCC: {bests[0]:.4f}
                PLCC:  {bests[1]:.4f}
                KROCC: {bests[2]:.4f}
                RMSE:  {bests[3]:.4f}."""
            )

            print(
                f"""
                the best validation accuracy of the model-n is as follows:
                SROCC: {bests_n[0]:.4f}
                PLCC:  {bests_n[1]:.4f}
                KROCC: {bests_n[2]:.4f}
                RMSE:  {bests_n[3]:.4f}."""
            )

        

       


if __name__ == "__main__":
    main()
