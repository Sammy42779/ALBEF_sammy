CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port 1994 VQA_mixgen.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/configs/VQA.yaml \
--output_dir /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/fine_tune_mixgen/VQA \
--checkpoint /data1/ld/checkpoint/ckpt_ALBEF/pre_train_14m/ALBEF.pth