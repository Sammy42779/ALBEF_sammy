#!/bin/sh
ckpt='/ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/fine_tune_aug/VE/checkpoint_best.pth'
out='/ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/fine_tune_aug/VE/IC'

# for c in 'gaussian_noise' 'shot_noise' 'impulse_noise' 'defocus_blur' 'glass_blur' 'motion_blur' 'zoom_blur' 'snow' 'frost' 'fog' 'brightness' 'contrast' 'elastic_transform' 'pixelate' 'jpeg_compression' 'speckle_noise'
for c in 'speckle_noise'
do
CUDA_VISIBLE_DEVICES=1,3,4,6,7 python -m torch.distributed.launch --nproc_per_node=5 --use_env --master_port 1994 VE_eval_IC.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/configs/VE.yaml \
--output_dir $out \
--checkpoint $ckpt \
--evaluate \
--corruption $c \
--severity s1

CUDA_VISIBLE_DEVICES=1,3,4,6,7 python -m torch.distributed.launch --nproc_per_node=5 --use_env --master_port 1994 VE_eval_IC.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/configs/VE.yaml \
--output_dir $out \
--checkpoint $ckpt \
--evaluate \
--corruption $c \
--severity s2

CUDA_VISIBLE_DEVICES=1,3,4,6,7 python -m torch.distributed.launch --nproc_per_node=5 --use_env --master_port 1994 VE_eval_IC.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/configs/VE.yaml \
--output_dir $out \
--checkpoint $ckpt \
--evaluate \
--corruption $c \
--severity s3

CUDA_VISIBLE_DEVICES=1,3,4,6,7 python -m torch.distributed.launch --nproc_per_node=5 --use_env --master_port 1994 VE_eval_IC.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/configs/VE.yaml \
--output_dir $out \
--checkpoint $ckpt \
--evaluate \
--corruption $c \
--severity s4


CUDA_VISIBLE_DEVICES=1,3,4,6,7 python -m torch.distributed.launch --nproc_per_node=5 --use_env --master_port 1994 VE_eval_IC.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/configs/VE.yaml \
--output_dir $out \
--checkpoint $ckpt \
--evaluate \
--corruption $c \
--severity s5
done
