#!/bin/bash
#SBATCH -o ./fine_tune/job.%j.out    # 脚本执行的输出将被保存在当job.%j.out文件下，%j表示作业号;
#SBATCH -p v100          # 作业提交的指定分区队列为v100;如果使用rtx，则为-p titan
#SBATCH --qos=v100       # 指定作业的QOS为v100; 如果使用titan，则为--qos=titan
#SBATCH -J VR_jq        # 作业在调度系统中的作业名为myFirstJob;
#SBATCH --nodes=1        # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1;
#SBATCH --ntasks-per-node=1 # 每个节点上运行一个任务，默认一情况下也可理解为每个节点使用一个核心,最大不能超过24;
#SBATCH --gres=gpu:4    # 指定作业的需要的GPU卡数量，最大不能超过4;

path_dir='/home/zhengf_lab/cse12032460/ld'

source activate base 

python -m torch.distributed.run --nproc_per_node=4 --master_port 1899 NLVR.py \
--config ${path_dir}/multimodal_models/ALBEF_sammy/configs/NLVR_jq.yaml \
--output_dir ${path_dir}/multimodal_models/ALBEF_sammy/fine_tune/VR \
--checkpoint ${path_dir}/data/checkpoint/ckpt_ALBEF/ckpt_VR_pretrain/pretrain_model_nlvr.pth