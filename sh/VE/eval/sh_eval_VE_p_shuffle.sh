CUDA_VISIBLE_DEVICES=2,4,6 python -m torch.distributed.launch --nproc_per_node=3 --use_env --master_port 1994 VE_eval_p_shuffle.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/configs/VE.yaml \
--output_dir /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/output/VE/p_shuffle \
--checkpoint /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/fine_tune/VE/checkpoint_best.pth \
--evaluate 