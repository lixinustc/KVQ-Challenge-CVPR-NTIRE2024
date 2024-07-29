


cd /data2/luyt/KSVQE

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port 3332 --use_env train_ddp.py --o config/Kwai_KSVQE.yml  --gpu_id 0,1,2,3 -r checkpoint_ddp/ > log/Kwai_KSVQE_ddp_loadpretrained.log 2>&1 &
