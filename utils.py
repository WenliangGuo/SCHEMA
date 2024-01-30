import os
import numpy as np
import pandas as pd
import clip
import json
import torch
import logging
import random
import torch.distributed as dist
import re

def parse_task_info(task_info_path):
    task_info = dict()
    with open(task_info_path, "r") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 6):
            task_info[lines[i].strip()] = {
                "name": lines[i+1].strip(),
                "url": lines[i+2].strip(),
                "num_steps": int(lines[i+3].strip()),
                "steps": lines[i+4].strip().split(","),          
            }
    return task_info


def parse_annotation(anot_dir, task_info, idices_mapping):
    annotation = dict()
    action_collection = idices_mapping["action_idx"]
    reduced_action_collection = idices_mapping["rd_action_idx"] 
    task_collection = idices_mapping["task_idx"]

    for file in os.listdir(anot_dir):
        info = pd.read_csv(os.path.join(anot_dir, file), header=None)
        v_name = file.split(".")[0]
        task_id = v_name[:v_name.find("_")]
        video_id = v_name[v_name.find("_")+1:]
        annotation[video_id] = []
        for i in range(len(info)):
            action_id = int(info.iloc[i][0])
            task = task_info[task_id]["name"].strip()
            action = task_info[task_id]["steps"][action_id-1].strip()

            whole_action_id = action_collection["{}_{}".format(task, action)]
            reduced_action_id = reduced_action_collection[action]
            task_nid = task_collection[task]

            annotation[video_id].append({
                "task": task,
                "task_id": task_nid,
                "action": action,
                "action_id": whole_action_id,
                "reduced_action_id": reduced_action_id,
                "start": int(np.round(float(info.iloc[i][1]))),
                "end": int(np.round(float(info.iloc[i][2]))),
            })

    return annotation

def crossstask_make_prompt_feature(clip_model, dsp_dir_json, action_collect, device, type="desc"):
    if type == "desc" or type == "action":
        with open(os.path.join(dsp_dir_json), "r") as f:
            description = json.load(f)

    s_feature = []
    e_feature = []
    a_feature = []

    with torch.no_grad():
        for n, (task_act, idx) in enumerate(action_collect.items()):
            task = task_act.split("_")[0]
            action = task_act.split("_")[1]
            if type == "desc":
                s_prompt = ["before " + dsp for dsp in description[task][action]["before"]] # before 0504
                e_prompt = ["after " + dsp for dsp in description[task][action]["after"]]
                s_prompt_token = clip.tokenize(s_prompt).to(device)
                e_prompt_token = clip.tokenize(e_prompt).to(device)
                s_prompt_features = clip_model.encode_text(s_prompt_token).cpu().numpy()
                e_prompt_features = clip_model.encode_text(e_prompt_token).cpu().numpy()
                s_feature.append(s_prompt_features)
                e_feature.append(e_prompt_features)
            
            elif type == "action":
                a_prompt = description[task][action]["description"]
                a_prompt_token = clip.tokenize(a_prompt).to(device)
                a_prompt_features = clip_model.encode_text(a_prompt_token).cpu().numpy()
                a_feature.append(a_prompt_features)

            elif type == "template_1":
                s_prompt = f"A photo indicating the state before the action of {action} for the goal of {task}"
                e_prompt = f"A photo indicating the state after the action of {action} for the goal of {task}"
                s_prompt_token = clip.tokenize(s_prompt).to(device)
                e_prompt_token = clip.tokenize(e_prompt).to(device)
                s_prompt_features = clip_model.encode_text(s_prompt_token).cpu().numpy()
                e_prompt_features = clip_model.encode_text(e_prompt_token).cpu().numpy()
                s_feature.append(s_prompt_features)
                e_feature.append(e_prompt_features)
            
            elif type == "template_2":
                s_prompt = f"The start state of the action of {action} for the goal of {task}"
                e_prompt = f"The end state of the action of {action} for the goal of {task}"
                s_prompt_token = clip.tokenize(s_prompt).to(device)
                e_prompt_token = clip.tokenize(e_prompt).to(device)
                s_prompt_features = clip_model.encode_text(s_prompt_token).cpu().numpy()
                e_prompt_features = clip_model.encode_text(e_prompt_token).cpu().numpy()
                s_feature.append(s_prompt_features)
                e_feature.append(e_prompt_features)

            elif type == "template_3":
                s_prompt = f"The start state of the action of {action}"
                e_prompt = f"The end state of the action of {action}"
                s_prompt_token = clip.tokenize(s_prompt).to(device)
                e_prompt_token = clip.tokenize(e_prompt).to(device)
                s_prompt_features = clip_model.encode_text(s_prompt_token).cpu().numpy()
                e_prompt_features = clip_model.encode_text(e_prompt_token).cpu().numpy()
                s_feature.append(s_prompt_features)
                e_feature.append(e_prompt_features)
            

    if type == "desc" or type == "template_1" or type == "template_2" or type == "template_3":
        s_feature = np.stack(s_feature)
        e_feature = np.stack(e_feature)
        state_feature = np.concatenate([s_feature, e_feature], axis = 1)
        return state_feature
    
    elif type == "action":
        a_feature = np.stack(a_feature)
        action_feature = a_feature
        return action_feature
    

def coin_make_prompt_feature(clip_model, dsp_dir_json, mapping_table, device, type="desc"):
    if type == "desc" or type == "action":
        with open(os.path.join(dsp_dir_json), "r") as f:
            description = json.load(f)

    dt_df = pd.read_excel(mapping_table, sheet_name='target_action_mapping')
    ## create a dictionary based on the taxonomy
    dt_list = []
    for i in range(len(dt_df)):
        task_id = dt_df['Target Id'][i]
        taeget_label = re.sub(r'([A-Z])', r' \1', dt_df['Target Label'][i]).strip()
        action_id = int(dt_df['Action Id'][i])
        action_label = dt_df['Action Label'][i]
        dt_list.append({'task_id': task_id, 'task_label': taeget_label, 'action_id': action_id-1, 'action_label': action_label})
    
    ## sort the list based on the action id
    dt_list = sorted(dt_list, key=lambda k: k['action_id'])

    s_feature = []
    e_feature = []
    a_feature = []
    with torch.no_grad():
        for i in range(len(dt_list)):
            task_label = dt_list[i]['task_label']
            action_label = dt_list[i]['action_label']

            if type == "desc":
                s_prompt = ["before " + dsp for dsp in description[task_label][action_label]["before"]]
                e_prompt = ["after " + dsp for dsp in description[task_label][action_label]["after"]]
                s_prompt_token = clip.tokenize(s_prompt).to(device)
                e_prompt_token = clip.tokenize(e_prompt).to(device)
                s_prompt_features = clip_model.encode_text(s_prompt_token).cpu().numpy()
                e_prompt_features = clip_model.encode_text(e_prompt_token).cpu().numpy()
                s_feature.append(s_prompt_features)
                e_feature.append(e_prompt_features)
            
            elif type == "action":
                a_prompt = description[task_label][action_label]["description"]
                a_prompt_token = clip.tokenize(a_prompt).to(device)
                a_prompt_features = clip_model.encode_text(a_prompt_token).cpu().numpy()
                a_feature.append(a_prompt_features)
            
            elif type == "template_1":
                s_prompt = f"A photo indicating the state before the action of {action_label} for the goal of {task_label}"
                e_prompt = f"A photo indicating the state after the action of {action_label} for the goal of {task_label}"
                s_prompt_token = clip.tokenize(s_prompt).to(device)
                e_prompt_token = clip.tokenize(e_prompt).to(device)
                s_prompt_features = clip_model.encode_text(s_prompt_token).cpu().numpy()
                e_prompt_features = clip_model.encode_text(e_prompt_token).cpu().numpy()
                s_feature.append(s_prompt_features)
                e_feature.append(e_prompt_features)
            
            elif type == "template_2":
                s_prompt = f"The start state of the action of {action_label} for the goal of {task_label}"
                e_prompt = f"The end state of the action of {action_label} for the goal of {task_label}"
                s_prompt_token = clip.tokenize(s_prompt).to(device)
                e_prompt_token = clip.tokenize(e_prompt).to(device)
                s_prompt_features = clip_model.encode_text(s_prompt_token).cpu().numpy()
                e_prompt_features = clip_model.encode_text(e_prompt_token).cpu().numpy()
                s_feature.append(s_prompt_features)
                e_feature.append(e_prompt_features)

    if type == "desc" or type == "template_1" or type == "template_2":
        s_feature = np.stack(s_feature)
        e_feature = np.stack(e_feature)
        state_feature = np.concatenate([s_feature, e_feature], axis = 1)
        return state_feature
    
    elif type == "action":
        a_feature = np.stack(a_feature)
        action_feature = a_feature
        return action_feature


def niv_make_prompt_feature(clip_model, dsp_dir_json, task_steps_rec, device, type="desc"):
    with open(os.path.join(dsp_dir_json), "r") as f:
        description = json.load(f)

    with open(task_steps_rec, "r") as f:
        task_steps = json.load(f)

    dt_list = []
    for task, task_info in task_steps.items():
        task_id = task_info['task_id']
        action_info = task_info['actions']
        for action, action_id in action_info.items():
            dt_list.append({'task_id': task_id, 
                            'task_label': task, 
                            'action_id': action_id, 
                            'action_label': action})
    
    ## sort the list based on the action id
    dt_list = sorted(dt_list, key=lambda k: k['action_id'])

    s_feature = []
    e_feature = []
    a_feature = []
    with torch.no_grad():
        for i in range(len(dt_list)):
            task_label = dt_list[i]['task_label']
            action_label = dt_list[i]['action_label']

            if type == "desc":
                s_prompt = ["before " + dsp for dsp in description[task_label][action_label]["before"]]
                e_prompt = ["after " + dsp for dsp in description[task_label][action_label]["after"]]
                s_prompt_token = clip.tokenize(s_prompt).to(device)
                e_prompt_token = clip.tokenize(e_prompt).to(device)
                s_prompt_features = clip_model.encode_text(s_prompt_token).cpu().numpy()
                e_prompt_features = clip_model.encode_text(e_prompt_token).cpu().numpy()
                s_feature.append(s_prompt_features)
                e_feature.append(e_prompt_features)
            
            elif type == "action":
                a_prompt = description[task_label][action_label]["description"]
                a_prompt_token = clip.tokenize(a_prompt).to(device)
                a_prompt_features = clip_model.encode_text(a_prompt_token).cpu().numpy()
                a_feature.append(a_prompt_features)
            
            elif type == "template_1":
                s_prompt = f"A photo indicating the state before the action of {action_label} for the goal of {task_label}"
                e_prompt = f"A photo indicating the state after the action of {action_label} for the goal of {task_label}"
                s_prompt_token = clip.tokenize(s_prompt).to(device)
                e_prompt_token = clip.tokenize(e_prompt).to(device)
                s_prompt_features = clip_model.encode_text(s_prompt_token).cpu().numpy()
                e_prompt_features = clip_model.encode_text(e_prompt_token).cpu().numpy()
                s_feature.append(s_prompt_features)
                e_feature.append(e_prompt_features)
            
            elif type == "template_2":
                s_prompt = f"The start state of the action of {action_label} for the goal of {task_label}"
                e_prompt = f"The end state of the action of {action_label} for the goal of {task_label}"
                s_prompt_token = clip.tokenize(s_prompt).to(device)
                e_prompt_token = clip.tokenize(e_prompt).to(device)
                s_prompt_features = clip_model.encode_text(s_prompt_token).cpu().numpy()
                e_prompt_features = clip_model.encode_text(e_prompt_token).cpu().numpy()
                s_feature.append(s_prompt_features)
                e_feature.append(e_prompt_features)

    if type == "desc" or type == "template_1" or type == "template_2":
        s_feature = np.stack(s_feature)
        e_feature = np.stack(e_feature)
        state_feature = np.concatenate([s_feature, e_feature], axis = 1)
        return state_feature
    
    elif type == "action":
        a_feature = np.stack(a_feature)
        action_feature = a_feature
        return action_feature
    

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def setup_seed(seed):
    random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True


def read_task_info(path):
    titles = {}
    urls = {}
    n_steps = {}
    steps = {}
    with open(path, "r") as f:
        idx = f.readline()
        while idx != "":
            idx = idx.strip()
            titles[idx] = f.readline().strip()
            urls[idx] = f.readline().strip()
            n_steps[idx] = int(f.readline().strip())
            steps[idx] = f.readline().strip().split(",")
            next(f)
            idx = f.readline()
    return {"title": titles, "url": urls, "n_steps": n_steps, "steps": steps}