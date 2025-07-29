import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from share import *

import gc
import argparse
import datetime
from omegaconf import OmegaConf

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset

from datasets.multigen20m import MultiGen20M
from datasets.multitask_HIA import MultiTask_HIA
from datasets.multi_task_scheduler import BatchSchedulerSampler
from datasets.dataset_collate import collate_fn
from cldm.logger import ImageLogger, CheckpointEveryNSteps
from cldm.model import create_model, load_state_dict
from cldm.hack import enable_sliced_attention

from contextlib import redirect_stdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args")
    # Dataset configs
    parser.add_argument("--dataroot", type=str, required=True, help='path to dataset')
    parser.add_argument("--drop_rate", type=float, default=0.3, help='drop rate for classifier-free guidance')
    # Model configs
    parser.add_argument("--config", type=str, required=True, help='path to model config file')
    parser.add_argument("--sd_ckpt", type=str, required=True, help='path to pretrained stable diffusion checkpoint')
    parser.add_argument("--cn_ckpt", type=str, required=True, help='path to pretrained controlnet checkpoint')
    # Training configs
    parser.add_argument("-n", "--name", type=str, help='experiment name')
    parser.add_argument("--lr", type=float, default=1e-5, help='learning rate')
    parser.add_argument("--bs", type=int, default=4, help='batchsize per device')
    parser.add_argument("--max_steps", type=int, default=700000, help='max training steps')
    parser.add_argument("--gradacc", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--precision", type=int, default=32, help='precision')
    parser.add_argument("--save_memory", action='store_true', default=False, help='save memory using sliced attention')
    parser.add_argument("--img_logger_freq", type=int, default=10000, help='img logger freq')
    parser.add_argument("--ckpt_logger_freq", type=int, default=10000, help='ckpt logger freq')
    # ckpt save configs
    parser.add_argument("--ckpt_save_dir", type=str, default="",  help='ckpt save at /media/data1')
    # HIA Finetune configs
    parser.add_argument("--Pretrain_Whole_ckpt", type=str, default="", help='path to Pretrain_Whole_ckpt')
    args = parser.parse_args()

    
    

    # Save memory
    if args.save_memory:
        enable_sliced_attention()

    # Construct Dataset
    conf = OmegaConf.load(args.config)
    tasks = conf.model.params.control_stage_config.params.tasks
    # dataset = ConcatDataset([
    #     MultiGen20M(
    #         path_json=os.path.join(args.dataroot, 'json_files', f'aesthetics_plus_all_group_{task}_all.json'),
    #         path_meta=args.dataroot, task=task, drop_rate=args.drop_rate,
    #     ) for task in tasks
    # ])
    dataset = ConcatDataset([
        MultiTask_HIA(
            path_json=os.path.join(args.dataroot, 'json_files', f'prompt_{task}.json'),
            task=task, drop_rate=args.drop_rate,
        ) for task in tasks
    ])
    dataloader = DataLoader(
        dataset=dataset, num_workers=16, batch_size=args.bs, persistent_workers=True, collate_fn=collate_fn,
        sampler=BatchSchedulerSampler(dataset=dataset, batch_size=args.bs, distributed=True, shuffle=True),
    )
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
    
    # ========================================================加载相同task pretrain过的模型，为了进行finetune(直接将Whole model中的U-Net、ControlNet、所有LoRAs加载到当前模型中)
    if args.Pretrain_Whole_ckpt != "":
        copied_keys, missing_keys = [], []
        whole_weights = load_state_dict(args.Pretrain_Whole_ckpt, location='cpu')
        for k in whole_weights:
            if k not in scratch_dict:
                missing_keys.append(k)
            else:
                scratch_dict[k] = whole_weights[k].clone()
                copied_keys.append(k)
        os.makedirs('./tmp', exist_ok=True)
        with open('./tmp/pretrain_missing_keys_Whole.txt', 'w') as f:
            f.write('\n'.join(missing_keys))
        with open('./tmp/pretrain_copied_keys_Whole.txt', 'w') as f:
            f.write('\n'.join(copied_keys))
            
        # Load scratch_dict to model
        model.load_state_dict(scratch_dict, strict=True)
        print(f"Successfully initialize SD & ControlNet & LoRAs from {args.Pretrain_Whole_ckpt}")
        del scratch_dict, whole_weights
    else:
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
        with open('./tmp/pretrain_missing_keys_sd.txt', 'w') as f:
            f.write('\n'.join(missing_keys))
        with open('./tmp/pretrain_copied_keys_sd.txt', 'w') as f:
            f.write('\n'.join(copied_keys))

        # Copy ControlNet weights to scratch_dict
        copied_keys, missing_keys = [], []
        control_weights = load_state_dict(args.cn_ckpt, location='cpu')
        for k in control_weights:
            if 'control_model' in k:
                if k not in scratch_dict:
                    missing_keys.append(k)
                else:
                    scratch_dict[k] = control_weights[k].clone()
                    copied_keys.append(k)
        with open('./tmp/pretrain_missing_keys_cn.txt', 'w') as f:
            f.write('\n'.join(missing_keys))
        with open('./tmp/pretrain_copied_keys_cn.txt', 'w') as f:
            f.write('\n'.join(copied_keys))

        # Load scratch_dict to model
        model.load_state_dict(scratch_dict, strict=True)
        print(f"Successfully initialize SD from {args.sd_ckpt}")
        print(f"Successfully initialize ControlNet from {args.cn_ckpt}")
        del scratch_dict, sd_weights, control_weights
    gc.collect()

    # Build Trainer
    logger_img = ImageLogger(batch_frequency=args.img_logger_freq)
    logger_checkpoint = CheckpointEveryNSteps(save_step_frequency=args.ckpt_logger_freq, ckpt_save_dir=args.ckpt_save_dir)
    if args.name is None:
        args.name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        
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
        strategy='ddp', accelerator='gpu', devices=-1, accumulate_grad_batches=args.gradacc, replace_sampler_ddp=False,
        max_steps=args.max_steps, precision=args.precision, callbacks=[logger_img, logger_checkpoint, checkpoint_callback],
        default_root_dir=os.path.join('runs', args.name),
    )

    # Train!
    trainer.fit(model, dataloader)
