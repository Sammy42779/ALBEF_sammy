CUDA_VISIBLE_DEVICES=0,7 python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port 1999 VE_mixgen.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/configs/VE.yaml \
--output_dir /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/fine_tune_mixgen/VE \
--checkpoint /data1/ld/checkpoint/ckpt_ALBEF/pre_train_14m/ALBEF.pth
