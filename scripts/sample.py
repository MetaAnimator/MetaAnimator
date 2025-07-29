import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from share import *

import cv2
import einops
import argparse

import torch
import numpy as np
from torch.utils.data import Subset

from annotator.util import HWC3, resize_image
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.cldm_ctrlora_pretrain import ControlPretrainLDM
from datasets.multigen20m import MultiGen20M
from datasets.custom_dataset import CustomDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args")
    # Dataset configs
    parser.add_argument("--dataroot", type=str, required=True, help='path to dataset')
    parser.add_argument("--multigen20m", action='store_true', default=False, help='use multigen20m dataset')
    # parser.add_argument("--task", type=str, choices=[
    #     'hed', 'canny', 'seg', 'depth', 'normal', 'openpose', 'hedsketch',
    #     'bbox', 'outpainting', 'inpainting', 'blur', 'grayscale',
    # ], help='task name')
    parser.add_argument("--task", type=str, choices=[
        "bg_outpainting", "canny", "depth", "DWpose", "fg_inpainting", "normal", "sapiens_seg", "blurred", "pose_HaMeR"
        ], help='task name')
    # Model configs
    parser.add_argument("--config", type=str, required=True, help='path to model config file')
    parser.add_argument("--ckpt", type=str, required=True, help='path to trained checkpoint')
    parser.add_argument("--bs", type=int, default=1, help='batchsize per device') # 之前的inference batch size默认是1 
    # Sampling configs
    parser.add_argument("--n_samples", type=int, default=10, help='number of samples')
    parser.add_argument("--save_dir", type=str, required=True, help='path to save samples')
    parser.add_argument("--ddim_steps", type=int, default=50, help='number of DDIM steps')
    parser.add_argument("--ddim_eta", type=float, default=0.0, help='DDIM eta')
    parser.add_argument("--strength", type=float, default=1.0, help='strength of controlnet')
    parser.add_argument("--cfg", type=float, default=7.5, help='unconditional guidance scale')
    parser.add_argument("--empty_prompt", action='store_true', default=False, help='experimental: use empty prompt')
    parser.add_argument("--inversion_image", type=str, default="", help='path to inversion_image')
    parser.add_argument("--same_intial_noise", action='store_true', help='same_intial_noise')
    parser.add_argument("--inversion_trgt_image", default=False, help='inversion_source_image')
    args = parser.parse_args()

    # Construct Dataset
    if args.multigen20m:
        dataset = MultiGen20M(
            path_json=os.path.join(args.dataroot, 'json_files', f'aesthetics_plus_all_group_{args.task}_all.json'),
            path_meta=args.dataroot, task=args.task, drop_rate=0.0, random_cropping=False,
        )
    else:
        dataset = CustomDataset(args.dataroot)
    if args.n_samples < len(dataset):
        dataset = Subset(dataset, range(args.n_samples))
    print('Dataset size:', len(dataset))

    # Construct Model
    model = create_model(args.config).cpu()
    if isinstance(model, ControlPretrainLDM):
        model.control_model.switch_lora(args.task)
    weights = load_state_dict(args.ckpt, location='cpu')
    model.load_state_dict(weights, strict=True)
    if isinstance(model, ControlPretrainLDM):
        model.control_model.switch_lora(args.task)
    model = model.cuda()
    print(f"Successfully load model ckpt from {args.ckpt}")

    # Construct DDIM Sampler
    ddim_sampler = DDIMSampler(model)

    # Sample
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'sample'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'control'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'img'), exist_ok=True)

    with torch.no_grad():
        for idx, item in enumerate(dataset):  # type: ignore
            print(f'Sample {idx + 1} image')
            img = ((item['jpg'] + 1.0) / 2.0 * 255.0).astype(np.uint8)
            img = resize_image(HWC3(img), 512)                                  # img: np.uint8, [0, 255]
            prompt = item['txt'] if not args.empty_prompt else ''
            control = (item['hint'] * 255.0).astype(np.uint8)
            control = resize_image(HWC3(control), 512)
            control = torch.from_numpy(control).float().cuda() / 255.0
            control = einops.rearrange(control, 'h w c -> c h w')
            control = control[None, ...]                                        # control: torch.float32, [0, 1]

            H, W, C = img.shape
            shape = (4, H // 8, W // 8)

            # sample with prompts
            cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt])], "task": args.task}
            un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([''])], "task": args.task}
            model.control_scales = [args.strength] * 13
            
            # ====================================================== 原始实现：直接进行sample
            if args.inversion_image == "" and not args.same_intial_noise and not args.inversion_trgt_image:
                samples, _ = ddim_sampler.sample(args.ddim_steps, args.bs, shape, cond,
                                                verbose=False, eta=args.ddim_eta,
                                                unconditional_guidance_scale=args.cfg,
                                                unconditional_conditioning=un_cond)  # samples.size() :  torch.Size([bs, 4, 64, 64])
            # ====================================================== 改进实现：每个batch内的sample都是基于相同的initial noise进行
            elif args.inversion_image == "" and args.same_intial_noise and not args.inversion_trgt_image:
                if 'noise' not in locals():
                    noise = torch.randn((args.bs, shape[0], shape[1], shape[2]), device=model.betas.device)
                    print(f"noise.size() : {noise.size()}")
                samples, _ = ddim_sampler.sample(args.ddim_steps, args.bs, shape, cond,
                                                verbose=False, eta=args.ddim_eta,
                                                x_T=noise,
                                                unconditional_guidance_scale=args.cfg,
                                                unconditional_conditioning=un_cond)  # samples.size() :  torch.Size([bs, 4, 64, 64])
            # ====================================================== 改进实现：用一张图的DDIM Inverison作为initial noise
            elif args.inversion_image != "":
                from ddim_inversion_TokenFlow import Preprocess
                if 'inversion_model' not in locals():
                    inversion_model = Preprocess(device="cuda", resolution=512)
                    # print("Inversion Image: ", args.inversion_image)
                if 'inv_latents' not in locals():
                    inv_latents = inversion_model.extract_latents(data_path=args.inversion_image, save_flage=False, num_steps=50, inversion_prompt="person")
                    # print("successfully get latents from inversion model")
                noise = inv_latents.float().cuda()
                samples, intermediates = ddim_sampler.sample(args.ddim_steps, args.bs, shape, cond,
                                                            verbose=False, eta=args.ddim_eta,
                                                            x_T=noise,
                                                            unconditional_guidance_scale=args.cfg,
                                                            unconditional_conditioning=un_cond)     # samples.size() :  torch.Size([bs, 4, 64, 64])
            # ====================================================== 改进实现：用每张图的trgt image的DDIM Inverison作为initial noise
            elif args.inversion_trgt_image:
                from ddim_inversion_TokenFlow import Preprocess
                if 'inversion_model' not in locals():
                    inversion_model = Preprocess(device="cuda", resolution=512)
                print("Inversion Image: ", item['target_filename'])
                # item['target_filename'] = item['target_filename'].replace("selected_short_frames", "selected_short_frames-Meta_Tasks_Data").replace("/frame_", "/images_fg_inpainting/frame_")
                inv_latents = inversion_model.extract_latents(data_path=item['target_filename'], save_flage=False, num_steps=50, inversion_prompt="person")
                # print("successfully get latents from inversion model")
                noise = inv_latents.float().cuda()
                samples, intermediates = ddim_sampler.sample(args.ddim_steps, args.bs, shape, cond,
                                                            verbose=False, eta=args.ddim_eta,
                                                            x_T=noise,
                                                            unconditional_guidance_scale=args.cfg,
                                                            unconditional_conditioning=un_cond)     # samples.size() :  torch.Size([bs, 4, 64, 64])
            # ====================================================== 改进实现：用每张图的trgt image的DDIM Inverison作为initial noise
            # elif args.inversion_trgt_image:
            #     from ddim_inversion_TokenFlow import Preprocess
            #     if 'inversion_model' not in locals():
            #         inversion_model = Preprocess(device="cuda", resolution=512)
            #     print("Inversion Image: ", item['target_filename'])
                
            #     x = img
            #     if len(x.shape) == 3:
            #         x = x[..., None]
            #     from einops import rearrange
            #     x = rearrange(x, 'b h w c -> b c h w')
            #     x = torch.from_numpy(x).to(model.device).float()
            #     encoder_posterior = model.encode_first_stage(x)
            #     z = encoder_posterior
            #     print("z.size() : ", z.size())
                
            #     inv_latents = inversion_model.extract_latents(data_path=item['target_filename'], save_flage=False, num_steps=50, inversion_prompt="person")
            #     # print("successfully get latents from inversion model")
            #     noise = inv_latents.float().cuda()
            #     print("noise.size() : ", noise.size())
            #     samples, intermediates = ddim_sampler.sample(args.ddim_steps, args.bs, shape, cond,
            #                                                 verbose=False, eta=args.ddim_eta,
            #                                                 x_T=noise,
            #                                                 unconditional_guidance_scale=args.cfg,
            #                                                 unconditional_conditioning=un_cond)     # samples.size() :  torch.Size([bs, 4, 64, 64])
                
            
            
            samples = model.decode_first_stage(samples)                     # samples: torch.float32, [-1, 1]

            # Save
            samples = (einops.rearrange(samples[0], 'c h w -> h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(args.save_dir, 'sample', f'{idx}.png'), samples[..., ::-1])

            cv2.imwrite(os.path.join(args.save_dir, 'img', f'{idx}.png'), img[..., ::-1])

            control = (einops.rearrange(control[0], 'c h w -> h w c') * 255.0).cpu().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(args.save_dir, 'control', f'{idx}.png'), control[..., ::-1])

            with open(os.path.join(args.save_dir, 'prompt.txt'), 'a') as f:
                print(prompt.strip(), file=f)

    print('Done')
