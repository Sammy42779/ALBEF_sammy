## the checkpoint after TA pre-training, which can be fine-tuned with the following steps.

CUDA_VISIBLE_DEVICES=2,4 python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port 1899 NLVR.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF/configs/NLVR.yaml \
--output_dir /ssd2/ld/ICCV2023/multimodal_models/ALBEF_sammy/fine_tune/VR \
--checkpoint /data1/ld/checkpoint/ckpt_ALBEF/ckpt_VR_pretrain/pretrain_model_nlvr.pth 