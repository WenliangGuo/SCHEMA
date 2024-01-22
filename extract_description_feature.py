import torch
import os
import numpy as np
import clip
import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', 
                    default='crosstask', type=str, 
                    help='dataset')
args = parser.parse_args()

def main():
    ## load clip model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, _ = clip.load("ViT-L/14@336px", device=device)
    clip_model.eval()
    vision_encoder = clip_model.visual
    for name, param in vision_encoder.named_parameters():
        param.requires_grad = False

    if args.dataset == 'crosstask':
        dsp_dir_json = "data/descriptors_crosstask.json"
        with open("data/crosstask_idices.json", "r") as f:
            idices_mapping = json.load(f)
        action_collect = idices_mapping["action_idx"]

        state_prompt_features = \
            crossstask_make_prompt_feature(clip_model, dsp_dir_json, 
                                           action_collect, device, type="desc")
        action_prompt_features = \
            crossstask_make_prompt_feature(clip_model, dsp_dir_json,
                                             action_collect, device, type="action")
    
    elif args.dataset == "coin":
        dsp_dir_json = "data/descriptors_coin.json"
        state_prompt_features = \
            coin_make_prompt_feature(clip_model, dsp_dir_json, 
                                    os.path.join(args.root_dir, "taxonomy.xlsx"), device, type="desc")
        action_prompt_features = \
            coin_make_prompt_feature(clip_model, dsp_dir_json, 
                                    os.path.join(args.root_dir, "taxonomy.xlsx"), device, type="action")
    
    elif args.dataset == "niv":
        dsp_dir_json = "data/descriptors_niv.json"
        state_prompt_features = \
            niv_make_prompt_feature(clip_model, dsp_dir_json, 
                                    "dataset/niv/niv_action_idx.json", device, type="desc")
        action_prompt_features = \
            niv_make_prompt_feature(clip_model, dsp_dir_json,
                                    "dataset/niv/niv_action_idx.json", device, type="action")
        
    np.save(f'data/state_description_features/{args.dataset}_state_prompt_features.npy', state_prompt_features)
    np.save(f'data/action_description_features/{args.dataset}_action_prompt_features.npy', action_prompt_features)

if __name__ == "__main__":
    main()