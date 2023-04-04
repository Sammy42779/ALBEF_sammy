CUDA_VISIBLE_DEVICES=0,5,6 python -m torch.distributed.launch --nproc_per_node=3 --use_env --master_port 1994 Retrieval_p_shuffle.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/configs/Retrieval_flickr.yaml \
--output_dir /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/output/ITR/p_shuffle \
--checkpoint /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/fine_tune/ITR/checkpoint_best.pth \
--evaluate 