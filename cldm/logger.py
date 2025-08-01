import os

import numpy as np
import torch
import torchvision
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only


class ImageLogger(Callback):
    """
    Adapted from https://github.com/lllyasviel/ControlNet/blob/main/cldm/logger.py
    """
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "gs-{:06}_e-{:06}_b-{:06}.png".format(global_step, current_epoch, batch_idx)
            path = os.path.join(root, k, filename)  # type: ignore
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, trainer, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(trainer.log_dir, split, images, pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx == 0 or (check_idx + 1) % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args):
        if not self.disabled:
            self.log_img(trainer, pl_module, batch, batch_idx, split="train")


class CheckpointEveryNSteps(Callback):
    """
    Adpated from https://github.com/salesforce/UniControl/blob/main/cldm/logger.py

    LICENSE of UniControl:
     * Copyright (c) 2023 Salesforce, Inc.
     * All rights reserved.
     * SPDX-License-Identifier: Apache License 2.0
     * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
     * By Can Qin
     * Modified from ControlNet repo: https://github.com/lllyasviel/ControlNet
     * Copyright (c) 2023 Lvmin Zhang and Maneesh Agrawala
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
        ckpt_save_dir=''
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
        self.ckpt_save_dir = ckpt_save_dir

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if self.check_frequency(global_step):
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            
            # ============================================因为存储空间不足做的改动
            if self.ckpt_save_dir != "":
                ckpt_path = os.path.join(self.ckpt_save_dir, ckpt_path)
            # print(f"\n\n Saving checkpoint at {ckpt_path} \n\n")
            # assert 0    
            # ============================================
            
            # print(f"Saving checkpoint at {ckpt_path}")
            trainer.save_checkpoint(ckpt_path)

    def check_frequency(self, check_idx):
        # return check_idx == 0 or (check_idx + 1) % self.save_step_frequency == 0
        return (check_idx + 1) % self.save_step_frequency == 0
