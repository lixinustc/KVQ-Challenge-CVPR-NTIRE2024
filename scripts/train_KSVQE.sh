
cd /data2/luyt/KSVQE
nohup python -u train.py  --o config/Kwai_KSVQE.yml -r checkpoint/ --gpu_id 0,1 > log/Kwai_KSVQE.log 2>&1 &

