CUDA_VISIBLE_DEVICES=0,3,4,7 python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 1899 VE.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF/configs/VE.yaml \
--output_dir /ssd2/ld/ICCV2023/multimodal_models/ALBEF/fine_tune/VE \
--checkpoint /data1/ld/checkpoint/ckpt_ALBEF/pre_train_14m/ALBEF.pth