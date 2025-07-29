# export CUDA_VISIBLE_DEVICES=7

# ======================== Sample MultiTask BaseCtrl&LoRA========================
CUDA_VISIBLE_DEVICES=0 python scripts/sample.py \
    --dataroot ./data/ctrlora_pretrain_sd15_9tasks_rank128_HIA_with_Temporal_Layers.yaml \
    --config ./configs/ctrlora_pretrain_sd15_9tasks_rank128_HIA.yaml \
    --ckpt /root/autodl-tmp/projects/MetAine/runs/exp_TikTok_9task-ExpID_8-finetune-video9-ExpID_4/lightning_logs/version_0/checkpoints/N-Step-Checkpoint_global_step=499.ckpt \
    --n_samples 300 \
    --bs 1 \
    --task DWPose \
    --save_dir ./runs/exp_TikTok_9task-ExpID_8-finetune-video9-ExpID_4/lightning_logs/version_0/samples \
    --Motion_Module_ckpt /root/autodl-tmp/projects/MetAine/ckpts/mm.ckpt \
    --inversion_image /root/autodl-tmp/projects/MetAine/TikTok/reference_rename/9.png
    # --same_intial_noise
