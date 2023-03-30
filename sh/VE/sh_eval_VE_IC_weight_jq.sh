#!/bin/bash
#SBATCH -o ./fine_tune_weight_kl/job.%j.out    # 脚本执行的输出将被保存在当job.%j.out文件下，%j表示作业号;
#SBATCH -p v100          # 作业提交的指定分区队列为v100;如果使用rtx，则为-p titan
#SBATCH --qos=v100       # 指定作业的QOS为v100; 如果使用titan，则为--qos=titan
#SBATCH -J VE_weight_kl_jq        # 作业在调度系统中的作业名为myFirstJob;
#SBATCH --nodes=1        # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1;
#SBATCH --ntasks-per-node=1 # 每个节点上运行一个任务，默认一情况下也可理解为每个节点使用一个核心,最大不能超过24;
#SBATCH --gres=gpu:4    # 指定作业的需要的GPU卡数量，最大不能超过4;

source activate base

path_dir='/home/zhengf_lab/cse30016037/ld'

ckpt=${path_dir}/multimodal_models/ALBEF_sammy/fine_tune_weight_kl/VE/checkpoint_best.pth
out=${path_dir}/multimodal_models/ALBEF_sammy/fine_tune_weight_kl/VE/IC

for c in 'gaussian_noise' 'shot_noise' 'impulse_noise' 'defocus_blur' 'glass_blur' 'motion_blur' 'zoom_blur' 'snow' 'frost' 'fog' 'brightness' 'contrast' 'elastic_transform' 'pixelate' 'jpeg_compression' 'speckle_noise'
do
CUDA_VISIBLE_DEVICES=0,1,2,7 python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 1999 VE_eval_IC.py \
--config ${path_dir}/multimodal_models/ALBEF_sammy/configs/VE.yaml \
--output_dir $out \
--checkpoint $ckpt \
--evaluate \
--corruption $c \
--severity s1

CUDA_VISIBLE_DEVICES=0,1,2,7 python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 1999 VE_eval_IC.py \
--config ${path_dir}/multimodal_models/ALBEF_sammy/configs/VE.yaml \
--output_dir $out \
--checkpoint $ckpt \
--evaluate \
--corruption $c \
--severity s2

CUDA_VISIBLE_DEVICES=0,1,2,7 python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 1999 VE_eval_IC.py \
--config ${path_dir}/multimodal_models/ALBEF_sammy/configs/VE.yaml \
--output_dir $out \
--checkpoint $ckpt \
--evaluate \
--corruption $c \
--severity s3

CUDA_VISIBLE_DEVICES=0,1,2,7 python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 1999 VE_eval_IC.py \
--config ${path_dir}/multimodal_models/ALBEF_sammy/configs/VE.yaml \
--output_dir $out \
--checkpoint $ckpt \
--evaluate \
--corruption $c \
--severity s4


CUDA_VISIBLE_DEVICES=0,1,2,7 python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 1999 VE_eval_IC.py \
--config ${path_dir}/multimodal_models/ALBEF_sammy/configs/VE.yaml \
--output_dir $out \
--checkpoint $ckpt \
--evaluate \
--corruption $c \
--severity s5
done
