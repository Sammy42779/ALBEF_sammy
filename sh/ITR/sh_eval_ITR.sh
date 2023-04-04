CUDA_VISIBLE_DEVICES=0,2,4 python -m torch.distributed.launch --nproc_per_node=3 --use_env --master_port 2000 Retrieval_eval.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF/configs/Retrieval_flickr.yaml \
--output_dir /ssd2/ld/ICCV2023/multimodal_models/ALBEF/reimp_output/Retrieval_flickr \
# --checkpoint /data1/ld/checkpoint/ckpt_ALBEF/ckpt_ITR_flickr30k/flickr30k.pth \
--checkpoint /ssd2/ld/ICCV2023/multimodal_models/ALBEF/fine_tune/Retrieval_flickr/checkpoint_best.pth \
--evaluate