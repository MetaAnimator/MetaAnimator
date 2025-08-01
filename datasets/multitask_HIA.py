"""
Adapted from https://github.com/salesforce/UniControl/blob/main/train_util/dataset.py

The License of UniControl is as follows:
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Can Qin
"""

import json
import cv2
import numpy as np

from torch.utils.data import Dataset
import random


class MultiTask_HIA(Dataset):
    def __init__(self, path_json, task, drop_rate=0.3, random_cropping=True):
        self.data = []
        self.task = task
        with open(path_json, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        if self.task not in ["bg_outpainting", "canny", "depth", "DWpose", "fg_inpainting", "normal", "sapiens_seg", "blurred", "pose_HaMeR"]:
            print('TASK NOT MATCH')
            return

        self.resolution = 512
        self.none_loop = 0
        self.drop_rate = drop_rate
        self.random_cropping = random_cropping

    def resize_image_control(self, control_image, resolution):
        H, W, C = control_image.shape
        if W >= H:
            crop = H
            if self.random_cropping:
                crop_l = random.randint(0, W-crop)  # 2nd value is inclusive
            else:
                crop_l = (W - crop) // 2
            crop_r = crop_l + crop
            crop_t = 0
            crop_b = H
        else:
            crop = W
            if self.random_cropping:
                crop_t = random.randint(0, H-crop)  # 2nd value is inclusive
            else:
                crop_t = (H - crop) // 2
            crop_b = crop_t + crop
            crop_l = 0
            crop_r = W
        control_image = control_image[crop_t: crop_b, crop_l:crop_r]
        H = float(H)
        W = float(W)
        k = float(resolution) / min(H, W)
        img = cv2.resize(control_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        return img, [crop_t/H, crop_b/H, crop_l/W, crop_r/W]

    def resize_image_target(self, target_image, resolution, sizes):
        H, W, C = target_image.shape
        crop_t_rate, crop_b_rate, crop_l_rate, crop_r_rate = sizes[0], sizes[1], sizes[2], sizes[3]
        crop_t, crop_b, crop_l, crop_r = int(crop_t_rate*H), int(crop_b_rate*H), int(crop_l_rate*W), int(crop_r_rate*W)
        target_image = target_image[crop_t: crop_b, crop_l:crop_r]
        H = float(H)
        W = float(W)
        k = float(resolution) / min(H, W)
        img = cv2.resize(target_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_img = cv2.imread(item['source'])
        target_img = cv2.imread(item['target'])
        prompt = item['prompt']

        # source_img, sizes = self.resize_image_control(source_img, self.resolution)
        # target_img = self.resize_image_target(target_img, self.resolution, sizes)

        # Do not forget that OpenCV read images in BGR order.
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source_img = source_img.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target_img = (target_img.astype(np.float32) / 127.5) - 1.0

        prompt = prompt if random.uniform(0, 1) > self.drop_rate else ''  # dropout prompt
        
        if self.task == "DWpose_with_mask":
            mask_img = cv2.imread(item['mask'], cv2.IMREAD_GRAYSCALE)
            # Normalize mask images to [0, 1].
            mask_img = mask_img.astype(np.float32) / 255.0
            return dict(jpg=target_img, txt=prompt, hint=source_img, task=self.task, mask=mask_img)
        
        return dict(jpg=target_img, txt=prompt, hint=source_img, task=self.task)

def _test():
    dataset = MultiTask_HIA(path_json="/root/autodl-tmp/projects/MetAine/data/exp17_TikTok_demo/json_files/prompt_bg_outpainting.json", 
                            task="bg_outpainting", drop_rate=0.3)
    print(len(dataset))

    item = dataset[123]
    jpg = item['jpg']
    txt = item['txt']
    hint = item['hint']
    task = item['task']
    print(txt)
    print(jpg.shape)
    print(hint.shape)
    print(task)


if __name__ == '__main__':
    _test()
