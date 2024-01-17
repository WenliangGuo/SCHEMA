import torch
import os
import numpy as np
import clip
from utils import *
from tools.parser import create_parser

def main(args):
    ## load clip model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, _ = clip.load("ViT-L/14@336px", device=device)
    clip_model.eval()
    vision_encoder = clip_model.visual
    for name, param in vision_encoder.named_parameters():
        param.requires_grad = False

    if args.dataset == 'crosstask_clip' or args.dataset == 'crosstask_howto100m':
        anot_dir = os.path.join(args.root_dir, "annotations")
        task_info_path = os.path.join(args.root_dir, "tasks_primary.txt")
        task_info = parse_task_info(task_info_path)
        anot_info, action_collect = parse_annotation(anot_dir, task_info)

        # produce prompt feature
        state_prompt_features = \
            crossstask_make_prompt_feature(clip_model, args.dsp_dir_json, 
                                           action_collect, device, type="desc")
    
    elif args.dataset == "coin":
        state_prompt_features = \
            coin_make_prompt_feature(clip_model, args.dsp_dir_json, 
                                    os.path.join(args.root_dir, "taxonomy.xlsx"), device, type="desc")
    
    elif args.dataset == "niv":
        state_prompt_features, action_prompt_features = \
            niv_make_prompt_feature(clip_model, args.dsp_dir_json, 
                                    "dataset/niv/niv_action_idx.json", device, type="desc")
        
    np.save(f'./data/template_3_features/crosstask_state_prompt_features.npy', state_prompt_features)
    # np.save(f'./data/{args.dataset}_action_prompt_features.npy', action_prompt_features)

if __name__ == "__main__":
    args = create_parser()
    main(args)