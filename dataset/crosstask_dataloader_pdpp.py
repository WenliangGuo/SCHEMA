import numpy as np
import itertools
from torch.utils.data import Dataset
import torch
import json
import os
from collections import namedtuple

class CrossTaskDataset(Dataset):
    def __init__(
        self, 
        anot_info, 
        feature_dir, 
        prompt_features, 
        video_list, 
        horizon = 3, 
        num_action = 133, 
        dataset = "crosstask_howto100m",
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
        self.state_features = []
        self.load_data()
        if self.mode == "train":
            # self.transition_matrix += 100 # works for logs/2023-05-10-01-16-46_v0 
            self.transition_matrix = self.cal_transition(self.transition_matrix)
            self.state_features = np.stack(self.state_features, axis = 0)

    def cal_transition(self, matrix):
        '''
        input:
        matrix: [num_action, num_action]
        output:
        transition: [num_action, num_action]
        '''
        transition = matrix / np.sum(matrix, axis = 1, keepdims = True)
        return transition


    def load_data(self):
        with open(self.video_list, "r") as f:
            video_info_dict = json.load(f)
        
        for video_info in video_info_dict:
            # video_id = video_info["id"]["vid"]
            video_id = video_info["vid"]
            video_anot = self.anot_info[video_id]
            task_id = video_anot[0]["task_id"]
            task = video_anot[0]["task"]
            
            if self.dataset == "crosstask_clip":
                try:
                    saved_features = \
                        np.load(os.path.join(self.feature_dir, "{}.npy".format(video_id)))
                except:
                    continue
            elif self.dataset == "crosstask_howto100m":
                try:
                    saved_features = \
                        np.load(os.path.join(self.feature_dir, "{}_{}.npy".\
                                                format(self.task_info[task], video_id)), 
                                allow_pickle=True)["frames_features"]
                except:
                    continue

            ## update transition matrix
            if self.mode == "train":
                for i in range(len(video_anot)-1):
                    cur_action = video_anot[i]["action_id"]-1
                    next_action = video_anot[i+1]["action_id"]-1
                    self.transition_matrix[cur_action, next_action] += 1
                    s_time = video_anot[i]["start"]
                    e_time = video_anot[i]["end"]
                    if s_time < 0 or e_time >= saved_features.shape[0]:
                        continue
                    self.state_features.append(saved_features[s_time])
                    self.state_features.append(saved_features[e_time])

            for i in range(len(video_anot)-self.horizon+1):
                all_features = []
                all_action_ids = []
                all_reduced_action_ids = []

                for j in range(self.horizon):
                    cur_video_anot = video_anot[i+j]
                    cur_action_id = cur_video_anot["action_id"]-1
                    cur_reduced_action_id = cur_video_anot["reduced_action_id"]-1

                    s_time = cur_video_anot["start"]
                    e_time = cur_video_anot["end"]

                    s_time = max(0, s_time)
                    if s_time + self.M + 1 <= len(saved_features):
                        start_feature = saved_features[s_time: s_time + self.M + 1]
                    else:
                        start_feature = saved_features[len(saved_features) - self.M - 1: len(saved_features)]
                    
                    # all_features.append(np.concatenate(start_feature))
                    if self.M == 0:
                        start_feature = np.expand_dims(start_feature, axis = 0)
                        all_features.append(start_feature)
                    else:
                        all_features.append(np.mean(start_feature, axis = 0))
                    all_action_ids.append(cur_action_id)
                    all_reduced_action_ids.append(cur_reduced_action_id)
                
                e_time = max(2, e_time)
                if e_time + self.M - 1 <= len(saved_features):
                    end_feature = saved_features[e_time - self.M:e_time + self.M - 1]
                else:
                    end_feature = saved_features[len(saved_features) - self.M - 1: len(saved_features)]

                # all_features.append(np.concatenate(end_feature))
                if self.M == 0:
                    end_feature = np.expand_dims(end_feature, axis = 0)
                    all_features.append(end_feature)
                else:
                    all_features.append(np.mean(end_feature, axis = 0))

                all_features = np.stack(all_features) 
                all_action_ids = np.array(all_action_ids)
                all_reduced_action_ids = np.array(all_reduced_action_ids)
                task_id = np.array(cur_video_anot["task_id"])
                
                self.data.append({"states": all_features,
                                "actions": all_reduced_action_ids, 
                                "init_actions": all_action_ids,
                                "tasks": task_id})
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        states = torch.as_tensor(self.data[idx]["states"], dtype=torch.float32)
        init_actions = torch.as_tensor(self.data[idx]["init_actions"], dtype=torch.long)
        reduced_actions = torch.as_tensor(self.data[idx]["actions"], dtype=torch.long)
        tasks = torch.as_tensor(self.data[idx]["tasks"], dtype=torch.long)
        # batch = Batch(states, init_actions, reduced_actions, tasks)

        # return states, init_actions, reduced_actions, tasks
        return states, init_actions, tasks



