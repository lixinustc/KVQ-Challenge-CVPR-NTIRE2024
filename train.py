import torch
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

from trainer import Trainer




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

    if not os.path.exists(args.resume):
                os.makedirs(args.resume)
   
        

    trainer=Trainer(args, opt )
    trainer.build_optimizer()
    
    for epoch in range(opt["num_epochs"]):
        print(f"End-to-end Epoch {epoch}:")
        
        bests,bests_n=trainer.train_eval_all_epoches(epoch)
        if opt["num_epochs"] >= 0:
            
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
