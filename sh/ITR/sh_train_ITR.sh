CUDA_VISIBLE_DEVICES=1,2,5,6 python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 1993 Retrieval.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF/configs/Retrieval_flickr.yaml \
--output_dir /ssd2/ld/ICCV2023/multimodal_models/ALBEF/fine_tune/Retrieval_flickr \
--checkpoint /data1/ld/checkpoint/ckpt_ALBEF/pre_train_14m/ALBEF.pth