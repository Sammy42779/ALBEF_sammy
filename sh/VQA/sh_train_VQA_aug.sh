CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port 1994 VQA_aug.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/configs/VQA.yaml \
--output_dir /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/fine_tune_aug/VQA \
--checkpoint /data1/ld/checkpoint/ckpt_ALBEF/pre_train_14m/ALBEF.pth