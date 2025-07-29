import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from share import *

import gc
import argparse
import datetime

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from datasets.multigen20m import MultiGen20M
from datasets.custom_dataset import CustomDataset, CustomDataset_2LoRA
from cldm.logger import ImageLogger, CheckpointEveryNSteps
from cldm.model import create_model, load_state_dict
from cldm.hack import enable_sliced_attention
import io
from contextlib import redirect_stdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args")
    # Dataset configs
    parser.add_argument("--dataroot", type=str, required=True, help='path to dataset')
    parser.add_argument("--drop_rate", type=float, default=0.3, help='drop rate for classifier-free guidance')
    parser.add_argument("--multigen20m", action='store_true', default=False, help='use multigen20m dataset')
    parser.add_argument("--task", type=str, choices=[
        'hed', 'canny', 'seg', 'depth', 'normal', 'openpose', 'hedsketch',
        'bbox', 'outpainting', 'inpainting', 'blur', 'grayscale',
    ], help='task name')
    parser.add_argument("--subset", type=int, default=0, help='train on a subset of the dataset')
    # Model configs
    parser.add_argument("--config", type=str, required=True, help='path to model config file')
    parser.add_argument("--sd_ckpt", type=str, required=True, help='path to pretrained stable diffusion checkpoint')
    parser.add_argument("--cn_ckpt", type=str, required=True, help='path to pretrained controlnet checkpoint')
    # Training configs
    parser.add_argument("-n", "--name", type=str, help='experiment name')
    parser.add_argument("--lr", type=float, default=1e-5, help='learning rate')
    parser.add_argument("--bs", type=int, default=1, help='batchsize per device')
    parser.add_argument("--max_steps", type=int, default=100000, help='max training steps')
    parser.add_argument("--gradacc", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--precision", type=int, default=32, help='precision')
    parser.add_argument("--save_memory", action='store_true', default=False, help='save memory using sliced attention')
    parser.add_argument("--img_logger_freq", type=int, default=1000, help='img logger freq')
    parser.add_argument("--ckpt_logger_freq", type=int, default=1000, help='ckpt logger freq')
    # HIA Finetune configs
    parser.add_argument("--TwoLoRA_finetune", action='store_true', default=False, help='2_LoRA_fintuen')
    parser.add_argument("--VAE_Fusion", action='store_true', default=False, help='multi input for single VAE')
    parser.add_argument("--One_Out_Of_TwoLoRA_finetune", action='store_true', default=False, help='One_Out_Of_TwoLoRA_finetune')
    parser.add_argument("--First_LoRA_ckpt", type=str, default="", help='path to 1st_LoRA_ckpt')
    parser.add_argument("--Pretrain_LoRA_ckpt", type=str, default="", help='path to Pretrain_LoRA_ckpt')
    # Temporal Layer configs
    parser.add_argument("--Motion_Module_ckpt", type=str, default="", help='path to Motion_Module_ckpt')
    args = parser.parse_args()

    # Save memory
    if args.save_memory:
        enable_sliced_attention()

    # Construct Dataset
    if args.multigen20m:
        dataset = MultiGen20M(
            path_json=os.path.join(args.dataroot, 'json_files', f'aesthetics_plus_all_group_{args.task}_all.json'),
            path_meta=args.dataroot, task=args.task, drop_rate=args.drop_rate,
        )
    elif args.TwoLoRA_finetune or args.VAE_Fusion:
        dataset = CustomDataset_2LoRA(args.dataroot, drop_rate=args.drop_rate)
    else:
        dataset = CustomDataset(args.dataroot, drop_rate=args.drop_rate)
    if args.subset > 0:
        dataset = Subset(dataset, range(args.subset))
    # 如果是训练temporal layer，这里改成shuffle=False是不是就要可以？因为我的prompt.json文件是按照顺序排列的
    dataloader = DataLoader(dataset, num_workers=16, batch_size=args.bs, shuffle=True) 
    print('Dataset size:', len(dataset))
    print('Number of devices:', torch.cuda.device_count())
    print('Batch size per device:', args.bs)
    print('Gradient accumulation:', args.gradacc)
    print('Total batch size:', args.bs * torch.cuda.device_count() * args.gradacc)
    # Construct Model
    model = create_model(args.config).cpu()
    model.learning_rate = args.lr
    model.sd_locked = True
    model.only_mid_control = False

    scratch_dict = model.state_dict()

    # Copy Stable Diffusion weights to scratch_dict
    copied_keys, missing_keys = [], []
    sd_weights = load_state_dict(args.sd_ckpt, location='cpu')
    for k in sd_weights:
        if k not in scratch_dict:
            missing_keys.append(k)
        else:
            scratch_dict[k] = sd_weights[k].clone()
            copied_keys.append(k)
    os.makedirs('./tmp', exist_ok=True)
    with open('./tmp/finetune_missing_keys_sd.txt', 'w') as f:
        f.write('\n'.join(missing_keys))
    with open('./tmp/finetune_copied_keys_sd.txt', 'w') as f:
        f.write('\n'.join(copied_keys))

    # Copy ControlNet weights to scratch_dict
    copied_keys, missing_keys = [], []
    control_weights = load_state_dict(args.cn_ckpt, location='cpu')
    for k in control_weights:
        if 'control_model' in k:
            if k not in scratch_dict:
                missing_keys.append(k)
            elif 'lora' in k:
                pass
            else:
                scratch_dict[k] = control_weights[k].clone()
                copied_keys.append(k)
    with open('./tmp/finetune_missing_keys_cn.txt', 'w') as f:
        f.write('\n'.join(missing_keys))
    with open('./tmp/finetune_copied_keys_cn.txt', 'w') as f:
        f.write('\n'.join(copied_keys))
    
    # ====================================== 载入训练好的LoRA模型，用于初始化
    if args.Pretrain_LoRA_ckpt != "":
        copied_keys, missing_keys = [], []
        lora_weights = load_state_dict(args.Pretrain_LoRA_ckpt, location='cpu')
        for k in lora_weights:
            if 'control_model' in k:
                if 'lora' in k or "zero_convs" in k or "middle_block_out" in k or "norm" in k:
                    if k not in scratch_dict:
                        missing_keys.append(k)
                    else:
                        scratch_dict[k] = lora_weights[k].clone()
                        copied_keys.append(k)
        os.makedirs('./tmp', exist_ok=True)
        with open('./tmp/finetune_missing_keys_lora.txt', 'w') as f:
            f.write('\n'.join(missing_keys))
        with open('./tmp/finetune_copied_keys_lora.txt', 'w') as f:
            f.write('\n'.join(copied_keys))
        del lora_weights
        print(f"Successfully initialize LoRA from {args.Pretrain_LoRA_ckpt}")
    
    # ====================================== 别忘了control_model_add中的ControlNet模块也需要用训练好的ControlNet-base模型进行初始化
    if args.TwoLoRA_finetune:
        copied_keys, missing_keys = [], []
        for k in control_weights:  # 这里要注意：k是ControlNet-base中的key，没有control_model_add
            if 'control_model' in k:
                new_k = k.replace('control_model', 'control_model_add')
                if k not in scratch_dict:
                    missing_keys.append(new_k)
                elif 'lora' in k:
                    pass
                else:
                    scratch_dict[new_k] = control_weights[k].clone()
                    copied_keys.append(new_k)
        with open('./tmp/finetune_missing_keys_cn_add.txt', 'w') as f:
            f.write('\n'.join(missing_keys))
        with open('./tmp/finetune_copied_keys_cn_add.txt', 'w') as f:
            f.write('\n'.join(copied_keys))
            
    # ====================================== 
    # if args.TwoLoRA_finetune and args.One_Out_Of_TwoLoRA_finetune:
    if args.TwoLoRA_finetune and args.First_LoRA_ckpt != "":
        copied_keys, missing_keys = [], []
        lora_weights = load_state_dict(args.First_LoRA_ckpt, location='cpu')
        for k in lora_weights:
            if 'control_model' in k:
                if 'lora' in k or "zero_convs" in k or "middle_block_out" in k or "norm" in k:
                    if k not in scratch_dict:
                        missing_keys.append(k)
                    else:
                        scratch_dict[k] = lora_weights[k].clone()
                        copied_keys.append(k)
        os.makedirs('./tmp', exist_ok=True)
        with open('./tmp/finetune_missing_keys_lora.txt', 'w') as f:
            f.write('\n'.join(missing_keys))
        with open('./tmp/finetune_copied_keys_lora.txt', 'w') as f:
            f.write('\n'.join(copied_keys))
        del lora_weights
        print(f"Successfully initialize 1st LoRA from {args.First_LoRA_ckpt}")
        
    # ====================================== 载入animatediff中训练好的motion module模型，用于初始化
    # ctrlora中的motion module的weight keys和animatediff中的motion module的weight keys不一样，但是模型结构是一模一样的，直接按照顺序载入就行
    # 具体可以看这两个文件：motion_nodule_state_dict_keys-animatediff.txt 和 motion_nodule_state_dict_keys-ctrlora.txt
    scratch_dict_mm_keys = []
    for idx, k in enumerate(scratch_dict):
        if "motion_module" in k:
            scratch_dict_mm_keys.append(k)
            # print(idx, k, scratch_dict[k].shape)
    if args.Motion_Module_ckpt != "":
        copied_keys, missing_keys = [], []
        motion_weights = load_state_dict(args.Motion_Module_ckpt, location='cpu')
        for idx, k in enumerate(motion_weights):
            trgt_key = scratch_dict_mm_keys[idx] # 根据顺序找出scratch_dict中对应的key
            if scratch_dict[trgt_key].shape != motion_weights[k].shape:
                print(f"Shape mismatch: {trgt_key} {scratch_dict[trgt_key].shape} \nVS\n {k} {motion_weights[k].shape}")
                missing_keys.append(trgt_key)
                assert 0
            scratch_dict[trgt_key] = motion_weights[k].clone()
            copied_keys.append(trgt_key)
        os.makedirs('./tmp', exist_ok=True)
        with open('./tmp/finetune_missing_keys_motion.txt', 'w') as f:
            f.write('\n'.join(missing_keys))
        with open('./tmp/finetune_copied_keys_motion.txt', 'w') as f:
            f.write('\n'.join(copied_keys))
        del motion_weights, scratch_dict_mm_keys
        print(f"Successfully initialize Motion_Module from {args.Motion_Module_ckpt}")
    
    # Load scratch_dict to model
    model.load_state_dict(scratch_dict, strict=True)
    print(f"Successfully initialize SD from {args.sd_ckpt}")
    print(f"Successfully initialize ControlNet from {args.cn_ckpt}")
    del sd_weights, control_weights, scratch_dict
    gc.collect()

    # Build Trainer
    logger_img = ImageLogger(batch_frequency=args.img_logger_freq)
    logger_checkpoint = CheckpointEveryNSteps(save_step_frequency=args.ckpt_logger_freq)
    if args.name is None:
        args.name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # ======================================
    # 这里在callbacks中调用了logger_img，log_images中调用了log_images方法（cldm.py中ControlLDM类中继承的方法）。
    # log_images方法会对模型进行ddim_sample，但是ddim_sample需要额外输入unconditional_conditioning。
    # 这里懒得再改了，所以就不调用logger_img就跳过这个问题了。
    # trainer = pl.Trainer(
    #     strategy='ddp', accelerator='gpu', devices=-1, accumulate_grad_batches=args.gradacc,
    #     max_steps=args.max_steps, precision=args.precision, callbacks=[logger_img, logger_checkpoint],
    #     default_root_dir=os.path.join('runs', args.name),
    # )
    # ======================================
    
    # ====================== 添加这个checkpoint_callback，是为了让pl.Trainer每隔一段时间保存一次checkpoint，而不是每次都保存，这样可以节省空间
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        # dirpath='./checkpoints',
        filename='{epoch}-{step}',
        save_top_k=0, # -1，0，1
        every_n_epochs=100,  # 每隔 5 个 epoch 保存一次模型
        save_weights_only=True
    )
    # 利用 **every_n_train_steps 、train_time_interval 、every_n_epochs **设置保存 checkpoint 的按照步数、时间、epoch数来保存 checkpoints，注意三者互斥，如果要同时实现对应的功能需要创建多个 MODELCHECKPOINT.
    # 利用 save_top_k 设置保存所有的模型，save_top_k 只能设置为 -1，0，1，分别表示保存所有的模型，不保存模型和保存最后一个模型
    
    trainer = pl.Trainer(
        strategy='ddp', accelerator='gpu', devices=-1, accumulate_grad_batches=args.gradacc,
        max_steps=args.max_steps, precision=args.precision, callbacks=[logger_checkpoint, checkpoint_callback],
        default_root_dir=os.path.join('runs', args.name),
    )

    # Train!
    trainer.fit(model, dataloader)
