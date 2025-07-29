import os
import sys
import argparse
import torch


def extract_weight(ckpt, key_words, exclude_key_words):
    # key_words是要包含的关键字，exclude_key_words是要排除的关键字
    if os.path.exists('keys_copied.txt'):
        os.remove('keys_copied.txt')
    if os.path.exists('keys_missed.txt'):
        os.remove('keys_missed.txt')
    save_ckpt = {}
    for k in ckpt.keys():
        if exclude_key_words is None: # 如果没有排除关键字exclude_key_words
            if key_words in k:
                save_ckpt[k] = ckpt[k]
                with open('keys_copied.txt', 'a') as f:
                    f.write(k + '\n')
            else:
                with open('keys_missed.txt', 'a') as f:
                    f.write(k + '\n')
        else:
            if key_words in k and exclude_key_words not in k:
                save_ckpt[k] = ckpt[k]
                with open('keys_copied.txt', 'a') as f:
                    f.write(k + '\n')
            else:
                with open('keys_missed.txt', 'a') as f:
                    f.write(k + '\n')
    return save_ckpt

def get_state_dict(d):
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict

def main():
    # args = get_parser().parse_args()
    ckpt = load_state_dict(args.ckpt, location='cpu')
    
    # =======================================================在这里设定关键字
    save_ckpt = extract_weight(ckpt, key_words='motion_modules', exclude_key_words="pose_guider")
    # =======================================================
    
    torch.save(save_ckpt, args.save_path)
    print(f'Extracted weights saved to {args.save_path}')
    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default="/root/autodl-tmp/projects/ctrlora/runs/exp_MM/lightning_logs/version_0/checkpoints/N-Step-Checkpoint_global_step=9999.ckpt", help='path to trained checkpoint')
    parser.add_argument('--save_path', type=str, default="mm.ckpt", help='path to save extracted weights')
    args = parser.parse_args()
    
    main()
