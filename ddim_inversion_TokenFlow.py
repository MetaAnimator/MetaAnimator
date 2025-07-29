from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
# suppress partial model loading warning
logging.set_verbosity_error()

import os
from tqdm import tqdm
import torch
import torch.nn as nn
import argparse
from pathlib import Path
# from util import *
import torchvision.transforms as T
from PIL import Image
import random
import numpy as np


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class Preprocess(nn.Module):
    def __init__(self, device, resolution):
        super().__init__()
        self.device = device
        self.resolution = resolution

        print(f'[INFO] loading stable diffusion...')
        model_key = "/root/.cache/huggingface/hub/models--sd-legacy--stable-diffusion-v1-5/snapshots/f03de327dd89b501a01da37fc5240cf4fdba85a1"
        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae",
                                                 torch_dtype=torch.float16).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder",
                                                          torch_dtype=torch.float16).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet",
                                                   torch_dtype=torch.float16).to(self.device)
        
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        # self.unet.enable_xformers_memory_efficient_attention()
        print(f'[INFO] loaded stable diffusion!')
        
        
        
    
    def get_data(self, frames_path): # frames_path = single image path
        # load frames
        paths =  [frames_path]
        frames = [Image.open(path).convert('RGB') for path in paths]
        frames = [frame.resize((self.resolution, self.resolution), resample=Image.Resampling.LANCZOS) for frame in frames]
        frames = torch.stack([T.ToTensor()(frame) for frame in frames]).to(torch.float16).to(self.device)
        # encode to latents
        latents = self.encode_imgs(frames, deterministic=True).to(torch.float16).to(self.device)
        return frames, latents
    
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        latents_batch = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents_batch).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs, deterministic=True):
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latent = posterior.mean if deterministic else posterior.sample()
        return latent * 0.18215

    @torch.no_grad()
    def ddim_inversion(self, cond, latent_frames, save_flage, save_path=""):
        timesteps = reversed(self.scheduler.timesteps)
        for i, t in enumerate(tqdm(timesteps)):
            x_batch = latent_frames
            model_input = x_batch
            cond_batch = cond.repeat(x_batch.shape[0], 1, 1)
                                                                
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[timesteps[i - 1]]
                if i > 0 else self.scheduler.final_alpha_cumprod
            )

            mu = alpha_prod_t ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma = (1 - alpha_prod_t) ** 0.5
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

            eps = self.unet(model_input, t, encoder_hidden_states=cond_batch).sample
            pred_x0 = (x_batch - sigma_prev * eps) / mu_prev
            latent_frames = mu * pred_x0 + sigma * eps
        if save_flage:
            torch.save(latent_frames, os.path.join(save_path, 'latents', f'noisy_latents_{t}.pt'))
        return latent_frames

    @torch.no_grad()
    def ddim_sample(self, x, cond):
        timesteps = self.scheduler.timesteps
        for i, t in enumerate(tqdm(timesteps)):
            x_batch = x
            model_input = x_batch
            cond_batch = cond.repeat(x_batch.shape[0], 1, 1)
            
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[timesteps[i + 1]]
                if i < len(timesteps) - 1
                else self.scheduler.final_alpha_cumprod
            )
            mu = alpha_prod_t ** 0.5
            sigma = (1 - alpha_prod_t) ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

            eps = self.unet(model_input, t, encoder_hidden_states=cond_batch).sample

            pred_x0 = (x_batch - sigma * eps) / mu
            x = mu_prev * pred_x0 + sigma_prev * eps
        return x

    @torch.no_grad()
    def extract_latents(self, 
                        data_path,
                        save_flage=True,
                        num_steps=50,
                        inversion_prompt='',
                        reconstruction_flag=False):
        self.scheduler.set_timesteps(num_steps)
        cond = self.get_text_embeds(inversion_prompt, "")[1].unsqueeze(0)
        
        self.frames, self.latents = self.get_data(data_path)
        latent_frames = self.latents

        inverted_x = self.ddim_inversion(cond,
                                         latent_frames,
                                         save_flage,)
        
        if reconstruction_flag:
            latent_reconstruction = self.ddim_sample(inverted_x, cond)                   
            rgb_reconstruction = self.decode_latents(latent_reconstruction)
            return rgb_reconstruction
        else:
            return inverted_x


def main(data_path="data/woman-running/00000.png", resolution=512, save_dir="demo_output", steps=50, inversion_prompt="", device="cuda"):
    seed_everything(1)
    save_path = os.path.join(save_dir,
                             Path(data_path).stem,
                             f'steps_{steps}') 
    os.makedirs(os.path.join(save_path, f'latents'), exist_ok=True)
    model = Preprocess(device, resolution)
    recon_frames = model.extract_latents(data_path,
                                         num_steps=steps,
                                         inversion_prompt=inversion_prompt,
                                         reconstruction_flag=True)

    if not os.path.isdir(os.path.join(save_path, f'frames')):
        os.mkdir(os.path.join(save_path, f'frames'))
    for i, frame in enumerate(recon_frames):
        T.ToPILImage()(frame).save(os.path.join(save_path, f'frames', f'{i:05d}.png'))

if __name__ == "__main__":
    main()