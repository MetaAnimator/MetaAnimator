
# ====================================================== 如果模型只有一个lora，用这个命令来保存
# python scripts/tool_extract_weights.py -t lora \
#     --ckpt $ROOT_DIR/runs/exp_TikTok-ExpID_12/lightning_logs/version_0/checkpoints/N-Step-Checkpoint_epoch=10_global_step=49999.ckpt \
#     --save_path /root/autodl-tmp/projects/ctrlora/ckpts/exp_models/exp_TikTok-ExpID_12-step=49999-lora_rank128.ckpt

# ====================================================== 如果模型是用多个lora训练的，用这个脚本来保存，可以将每个task对应的lora都保存下来
python scripts/tool_extract_weights.py -t lora \
    --ckpt $ROOT_DIR/runs/exp_TikTok_9task-ExpID_8/lightning_logs/version_0/checkpoints/N-Step-Checkpoint_epoch=10_global_step=49999.ckpt \
    --save_path /root/autodl-tmp/projects/ctrlora/ckpts/exp_models/exp_TikTok_9task-ExpID_8-step=49999-lora_rank128/DWpose_lora.ckpt \
    --from_base \
    --from_base_config ./configs/ctrlora_pretrain_sd15_9tasks_rank128_HIA.yaml