export CUDA_VISIBLE_DEVICES=0,1

# sd_ckpt: v1-5-pruned.ckpt  ,  realisticVisionV60B1_v51VAE.ckpt
# ======================== Train Temporal Layers ========================
python scripts/train_ctrlora_finetune.py \
    --dataroot ./data/exp_TikTok_9task-Motion_Module \
    --config ./configs/ctrlora_finetune_sd15_rank128_with_Temporal_Layers.yaml \
    --sd_ckpt ./ckpts/sd15/v1-5-pruned.ckpt \
    --cn_ckpt ./ckpts/exp_models/exp_TikTok_9task-ExpID_8-step=49999-cn.ckpt \
    --Pretrain_LoRA_ckpt ./ckpts/exp_models/exp_TikTok_9task-ExpID_8-step=49999-lora_rank128/DWpose_lora.ckpt \
    --name exp_TikTok_9task-Motion_Module \
    --bs 8 \
    --max_steps 10000 \
    --img_logger_freq 2000 \
    --ckpt_logger_freq 2000 \
    --Motion_Module_ckpt /root/autodl-tmp/projects/MetAine/ckpts/v3_sd15_mm.ckpt

