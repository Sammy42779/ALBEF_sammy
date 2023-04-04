# CUDA_VISIBLE_DEVICES=1,0,3 python -m torch.distributed.launch --nproc_per_node=3 --use_env --master_port 1993 Retrieval_aug.py \
# --config /ssd2/ld/ICCV2023/multimodal_models/ALBEF/configs/Retrieval_flickr.yaml \
# --output_dir /ssd2/ld/ICCV2023/multimodal_models/ALBEF/fine_tune_aug/Retrieval_flickr \
# --checkpoint /data1/ld/checkpoint/ckpt_ALBEF/pre_train_14m/ALBEF.pth \
# --no-jsd


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 1993 Retrieval_aug.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF/configs/Retrieval_flickr.yaml \
--output_dir /ssd2/ld/ICCV2023/multimodal_models/ALBEF/fine_tune_aug/Retrieval_flickr \
--checkpoint /data1/ld/checkpoint/ckpt_ALBEF/pre_train_14m/ALBEF.pth \
--no-jsd