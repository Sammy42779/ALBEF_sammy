#!/bin/sh
# ckpt='/ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/fine_tune_weight/VE/checkpoint_best.pth'
# out='/ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/fine_tune_weight/VE/IC'

# ckpt='/ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/fine_tune_weight_kl/VE/checkpoint_best.pth'
# out='/ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/fine_tune_weight_kl/VE/IC'

# ckpt='/ssd3/ld/fine_tune_weight_gair_kl/VE/checkpoint_best.pth'
# out='/ssd3/ld/fine_tune_weight_gair_kl/VE/IC'

### FOCAL LOSS
ckpt='/data1/ld/weight_aware_vlp_ft_ckpt/fine_tune_focal_loss/VE/checkpoint_best.pth'
out='/ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/fine_tune_weight_aware/focal_loss/VE/IC'

for c in 'gaussian_noise' 'shot_noise' 'impulse_noise' 'defocus_blur' 'glass_blur' 'motion_blur' 'zoom_blur' 'snow' 'frost' 'fog' 'brightness' 'contrast' 'elastic_transform' 'pixelate' 'jpeg_compression' 'speckle_noise'
do
CUDA_VISIBLE_DEVICES=1,4,6 python -m torch.distributed.launch --nproc_per_node=3 --use_env --master_port 1997 VE_eval_IC.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/configs/VE.yaml \
--output_dir $out \
--checkpoint $ckpt \
--evaluate \
--corruption $c \
--severity s1

CUDA_VISIBLE_DEVICES=1,4,6 python -m torch.distributed.launch --nproc_per_node=3 --use_env --master_port 1997 VE_eval_IC.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/configs/VE.yaml \
--output_dir $out \
--checkpoint $ckpt \
--evaluate \
--corruption $c \
--severity s2

CUDA_VISIBLE_DEVICES=1,4,6 python -m torch.distributed.launch --nproc_per_node=3 --use_env --master_port 1997 VE_eval_IC.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/configs/VE.yaml \
--output_dir $out \
--checkpoint $ckpt \
--evaluate \
--corruption $c \
--severity s3

CUDA_VISIBLE_DEVICES=1,4,6 python -m torch.distributed.launch --nproc_per_node=3 --use_env --master_port 1997 VE_eval_IC.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/configs/VE.yaml \
--output_dir $out \
--checkpoint $ckpt \
--evaluate \
--corruption $c \
--severity s4


CUDA_VISIBLE_DEVICES=1,4,6 python -m torch.distributed.launch --nproc_per_node=3 --use_env --master_port 1997 VE_eval_IC.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/configs/VE.yaml \
--output_dir $out \
--checkpoint $ckpt \
--evaluate \
--corruption $c \
--severity s5
done
