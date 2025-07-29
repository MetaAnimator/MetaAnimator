# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0

# ======================== Pretrain (with smaller drop rate) ========================
python scripts/train_ctrlora_pretrain.py \
    --dataroot ./data/exp_TikTok_9task-ExpID_8-finetune-video10 \
    --config ./configs/ctrlora_pretrain_sd15_9tasks_rank128_HIA.yaml \
    --sd_ckpt ./ckpts/sd15/v1-5-pruned.ckpt \
    --cn_ckpt ./ckpts/control_sd15_init.pth \
    --name exp_TikTok_9task-ExpID_8-finetune-video10-ExpID_0 \
    --bs 8 \
    --drop_rate 0.01 \
    --max_steps 500 \
    --img_logger_freq 100 \
    --ckpt_logger_freq 100 \

# tensorboard --logdir=./runs/exp_TikTok_9task-ExpID_12/lightning_logs/version_0