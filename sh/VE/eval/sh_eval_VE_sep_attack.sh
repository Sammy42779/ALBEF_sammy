## FOCAL LOSS
CUDA_VISIBLE_DEVICES=1,6 python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port 1994 VE_eval_sep_attack.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF/configs/VE.yaml \
--output_dir /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/fine_tune_weight_aware/focal_loss/VE/adv_sep_attack \
--checkpoint /data1/ld/weight_aware_vlp_ft_ckpt/fine_tune_focal_loss/VE/checkpoint_best.pth \
--evaluate