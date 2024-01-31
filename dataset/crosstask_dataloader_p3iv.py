import numpy as np
import itertools
from torch.utils.data import Dataset
import torch
import os
import pickle

class CrossTaskDataset(Dataset):
    def __init__(
        self, 
        anot_info, 
        feature_dir, 
        prompt_features, 
        video_list,
        horizon = 3, 
        num_action = 133,
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
        self.mode = mode
        self.M = M

        self.vid_gather = {}
        
        with open(video_list, "rb") as f:
            self.datasplit = pickle.load(f)
        self.datasplit = self.datasplit["train"] if mode == "train" else self.datasplit["test"]

        self.num_action = num_action
        self.transition_matrix = np.zeros((num_action, num_action))

        self.data = []
        self.load_data()
        if self.mode == "train":
            self.transition_matrix = self.cal_transition()

    def cal_transition(self):
        '''
        input:
        matrix: [num_action, num_action]
        output:
        transition: [num_action, num_action]
        '''

        matrix = np.zeros((self.num_action, self.num_action))
        for data in self.data:
            actions = data["actions"]
            adj = tuple(zip(actions[:-1], actions[1:]))
            # update transition matrix
            for i in range(len(adj)):
                matrix[adj[i]] += 1
        matrix += 100
        transition = matrix / np.sum(matrix, axis = 1, keepdims = True)
        return transition


    def load_data(self):
        ## gather all the indices of the same video
        for i in range(len(self.datasplit)):
            cur_vid = self.datasplit[i][1]
            if cur_vid not in self.vid_gather.keys():
                self.vid_gather[cur_vid] = []
            self.vid_gather[cur_vid].append(i)
        
        ## load data for each video
        for vid, idxs in self.vid_gather.items():
            npy_path = os.path.join(self.feature_dir, vid+".npy")
            if not os.path.exists(npy_path):
                continue
            saved_features = np.load(npy_path)
            video_anot = self.anot_info[vid]
            task_id = self.anot_info[vid][0]["task_id"]

            for idx in idxs:
                data_item = self.datasplit[idx]
                start_idx = data_item[4][0]
                end_idx = data_item[4][1]
                action_id = []
                state_feat = []


                for t in range(start_idx, end_idx):     ## end_idx - start_idx = time_horizon
                    action_id.append(video_anot[t]["action_id"]-1)
                    s_time = video_anot[t]["start"]
                    e_time = video_anot[t]["end"]
                    if s_time < 0 or e_time >= saved_features.shape[0]:
                        continue
                    ## smooth the features
                    s_offset_start = max(0, s_time-self.M//2)
                    s_offset_end = min(s_time+self.M//2+1,saved_features.shape[0])
                    e_offset_start = max(0, e_time-self.M//2)
                    e_offset_end = min(e_time+self.M//2+1,saved_features.shape[0])

                    start_feature = np.mean(saved_features[s_offset_start:s_offset_end], axis = 0)
                    end_feature = np.mean(saved_features[e_offset_start:e_offset_end], axis = 0)
                    state_feat.append(np.stack([start_feature, end_feature]))
                
                if len(state_feat) < self.horizon:
                    continue
                                
                self.data.extend([{"states": np.stack(state_feat),
                                   "actions": np.array(action_id), 
                                   "tasks": np.array(task_id)}])
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        states = self.data[idx]["states"]
        actions = self.data[idx]["actions"]
        tasks = self.data[idx]["tasks"]
        return torch.as_tensor(states, dtype=torch.float32), torch.as_tensor(actions, dtype=torch.long), torch.as_tensor(tasks, dtype=torch.long)