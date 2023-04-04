CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port 1994 VE_eval.py \
--config /ssd2/ld/ICCV2023/multimodal_models/ALBEF/configs/VE.yaml \
--output_dir /ssd2/ld/ICCV2023/multimodal_models/ALBEF/output/VE \
--checkpoint /data1/ld/checkpoint/ckpt_ALBEF/ckpt_VE_co_attack_reimp/ALBEF-VE.pth \
--evaluate