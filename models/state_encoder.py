import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.utils import img_text_similarlity, process_state_feat

class StateEncoder(nn.Module):
    def __init__(self, vis_input_dim, lang_input_dim, embed_dim, dropout=0.1, vis_mode=False, mid_state_align=False):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.state_encoder = nn.Sequential(
            nn.Linear(vis_input_dim, 2*vis_input_dim),
            nn.Linear(2*vis_input_dim, embed_dim)
        )
        self.desc_encoder = nn.Linear(lang_input_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.vis_mode = vis_mode
        self.mid_state_align = mid_state_align

    def forward(self, state_feat, all_state_desc_feat):
        '''
        input:
        state_feat:             (batch, time_horz, 2, input_dim)
        all_state_desc_feat:    (num_action, num_desc, input_dim)

        output:
        state_feat:             (batch, 2, embed_dim)
        inter_state_feat:       (batch, time_horz-1, embed_dim)
        state_logits:           (batch, 2, n_actions)
        state_desc_logits:      (batch, 2, n_desc)
        prompt_features:        (num_action, num_desc, embed_dim)
        '''

        batch_size = state_feat.shape[0]
        
        # pre-process state feats
        state_feat = process_state_feat(state_feat) # [batch, time_horz+1, input_dim]

        # encode description feature
        prompt_features = self.desc_encoder(self.dropout(all_state_desc_feat))
        n_actions = prompt_features.shape[0]
        num_desc = prompt_features.shape[1]
        start_desc_feat = prompt_features[:, :num_desc // 2, :]
        end_desc_feat   = prompt_features[:, num_desc // 2:, :]

        # encode state feature
        state_feat = self.state_encoder(self.dropout(state_feat))
        start_state_feat = state_feat[:, 0, :]  # [batch, embed_dim]
        end_state_feat = state_feat[:, -1, :]  # [batch, embed_dim]
        inter_state_feat = state_feat[:, 1:-1, :]  # [batch, time_horz-1, embed_dim]

        # contrast state and prompt feature
        if self.vis_mode is False:
            s_logits = img_text_similarlity(start_state_feat, start_desc_feat, self.logit_scale.exp())
            e_logits = img_text_similarlity(end_state_feat, end_desc_feat, self.logit_scale.exp())
            if self.mid_state_align is True:   #align mid-states when training
                mid_logits = []
                for k in range(inter_state_feat.shape[1]):
                    mid_e_logits = img_text_similarlity(inter_state_feat[:,k,:], end_desc_feat, self.logit_scale.exp())
                    mid_s_logits = img_text_similarlity(inter_state_feat[:,k,:], start_desc_feat, self.logit_scale.exp())
                    mid_logits.append(mid_e_logits.reshape(batch_size, n_actions, -1).mean(-1))
                    mid_logits.append(mid_s_logits.reshape(batch_size, n_actions, -1).mean(-1))
                mid_logits = torch.stack(mid_logits, 1) # [batch, 2*time_horz-2, n_actions]
   
        else:   # only for experimental visualization, usually not used
            self.logit_scale=nn.Parameter(torch.ones([]))
            s_logits = img_text_similarlity(start_state_feat, start_desc_feat, self.logit_scale)
            e_logits = img_text_similarlity(end_state_feat, end_desc_feat, self.logit_scale)

        state_desc_logits = torch.cat((s_logits.reshape(batch_size, -1).unsqueeze(1), 
                                    e_logits.reshape(batch_size, -1).unsqueeze(1)), 1) # [batch, 2, n_desc]
        
        s_logits = s_logits.reshape(batch_size, n_actions, -1).mean(-1)
        e_logits = e_logits.reshape(batch_size, n_actions, -1).mean(-1)
        if self.mid_state_align is True:
            state_logits = torch.cat((s_logits.unsqueeze(1), mid_logits, e_logits.unsqueeze(1)), 1)    # [batch, 2*time_horz, n_actions]
        else:
            state_logits = torch.cat((s_logits.unsqueeze(1), e_logits.unsqueeze(1)), 1) # [batch, 2, n_actions]

        state_feat = torch.cat([start_state_feat.unsqueeze(1), end_state_feat.unsqueeze(1)], 1)

        if self.vis_mode is False:
            return state_feat, inter_state_feat, state_logits, prompt_features
        else:
            return state_feat, inter_state_feat, state_logits, state_desc_logits, prompt_features
