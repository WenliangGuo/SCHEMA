import numpy as np
import itertools
from torch.utils.data import Dataset
import torch
import json
import os

class CrossTaskDataset(Dataset):
    def __init__(
        self, 
        anot_info, 
        feature_dir, 
        prompt_features, 
        video_list, 
        horizon = 3, 
        num_action = 133, 
        dataset = "crosstask",
        datasplit = 'base',
        aug_range = 0, 
        M = 2, 
        mode = "train", 
    ):
        super().__init__()
        self.anot_info = anot_info
        self.feature_dir = feature_dir
        self.prompt_features = prompt_features
        self.aug_range = aug_range
        self.horizon = horizon
        self.video_list = video_list
        self.max_duration = 0
        self.mode = mode
        self.M = M
        self.dataset = dataset
        self.datasplit = datasplit

        self.num_action = num_action
        self.transition_matrix = np.zeros((num_action, num_action))

        self.task_info = {"Make Jello Shots": 23521, 
                          "Build Simple Floating Shelves": 59684, 
                          "Make Taco Salad": 71781, 
                          "Grill Steak": 113766,
                          "Make Kimchi Fried Rice": 105222, 
                          "Make Meringue": 94276,
                          "Make a Latte": 53193, 
                          "Make Bread and Butter Pickles": 105253,
                          "Make Lemonade": 44047, 
                          "Make French Toast": 76400,
                          "Jack Up a Car": 16815, 
                          "Make Kerala Fish Curry": 95603,
                          "Make Banana Ice Cream": 109972, 
                          "Add Oil to Your Car": 44789,
                          "Change a Tire": 40567, 
                          "Make Irish Coffee": 77721,
                          "Make French Strawberry Cake": 87706, 
                          "Make Pancakes": 91515}

        self.data = []
        # self.state_features = []
        self.load_data()
        if self.mode == "train":
            # self.transition_matrix += 1 # works for logs/2023-05-10-01-16-46_v0 
            self.transition_matrix = self.cal_transition(self.transition_matrix)
            # self.state_features = np.stack(self.state_features, axis = 0)

    def cal_transition(self, matrix):
        '''
        input:
        matrix: [num_action, num_action]
        output:
        transition: [num_action, num_action]
        '''
        transition = matrix / np.sum(matrix, axis = 1, keepdims = True)
        return transition

    def select_prompts(self, actions):
        '''
        input:
        actions: [time_horz]
        output:
        cur_prompt_features: [2, num_prompts, embedding_dim]
        '''
        cur_prompt_features = []

        t = len(actions)-1
        cur_prompt_features.append(self.prompt_features[actions[0],:3,:])
        cur_prompt_features.append(self.prompt_features[actions[t],3:,:])

        cur_prompt_features = np.stack(cur_prompt_features, axis = 0)
        
        return cur_prompt_features

    def load_data(self):
        with open(self.video_list, "r") as f:
            video_info_dict = json.load(f)
        
        for video_info in video_info_dict:
            if self.datasplit == 'base':
                video_id = video_info["id"]["vid"]
            elif self.datasplit == 'pdpp':
                video_id = video_info["vid"]

            video_anot = self.anot_info[video_id]
            task_id = video_anot[0]["task_id"]
            task = video_anot[0]["task"]
            
            try:
                saved_features = \
                    np.load(os.path.join(self.feature_dir, "{}_{}.npy".\
                                            format(self.task_info[task], video_id)), 
                            allow_pickle=True)["frames_features"]
            except:
                continue
                        
            ## remove repeated actions
            # legal_video_anot = []
            # for i in range(len(video_anot)):
            #     if i == 0 or video_anot[i]["action_id"] != video_anot[i-1]["action_id"]:
            #         legal_video_anot.append(video_anot[i])
            # video_anot = legal_video_anot

            ## update transition matrix
            if self.mode == "train":
                for i in range(len(video_anot)-1):
                    cur_action = video_anot[i]["action_id"]-1
                    next_action = video_anot[i+1]["action_id"]-1
                    self.transition_matrix[cur_action, next_action] += 1


            for i in range(len(video_anot)-self.horizon+1):
                all_features = []
                all_action_ids = []

                for j in range(self.horizon):
                    ## Using adjacent frames for data augmentation
                    cur_video_anot = video_anot[i+j]
                    cur_action_id = cur_video_anot["action_id"]-1
                    features = []
                    
                    for frame_offset in range(-self.aug_range, self.aug_range+1):
                        s_time = cur_video_anot["start"]+frame_offset
                        e_time = cur_video_anot["end"]+frame_offset

                        if s_time < 0 or e_time >= saved_features.shape[0]:
                            continue
                        
                        s_offset_start = max(0, s_time-self.M//2)
                        s_offset_end = min(s_time+self.M//2+1,saved_features.shape[0])
                        e_offset_start = max(0, e_time-self.M//2)
                        e_offset_end = min(e_time+self.M//2+1,saved_features.shape[0])

                        start_feature = np.mean(saved_features[s_offset_start:s_offset_end], axis = 0)
                        end_feature = np.mean(saved_features[e_offset_start:e_offset_end], axis = 0)

                        features.append(np.stack((start_feature, end_feature)))

                    all_features.append(features)
                    all_action_ids.append(cur_action_id)

                task_id = cur_video_anot["task_id"]

                ## permutation of augmented features, action ids and prompts
                aug_features = itertools.product(*all_features)

                self.data.extend([{"states": np.stack(f),
                                   "actions": np.array(all_action_ids), 
                                   "tasks": np.array(task_id)} 
                                  for f in aug_features])
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        states = self.data[idx]["states"]
        actions = self.data[idx]["actions"]
        tasks = self.data[idx]["tasks"]
        return torch.as_tensor(states, dtype=torch.float32), torch.as_tensor(actions, dtype=torch.long), torch.as_tensor(tasks, dtype=torch.long)