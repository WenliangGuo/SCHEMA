import torch
import torch.nn as nn
import numpy as np
from models.utils import img_text_similarlity

class StateEncoder(nn.Module):
    def __init__(self, vis_input_dim, lang_input_dim, embed_dim, dropout=0.1):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.state_encoder = nn.Sequential(
            nn.Linear(vis_input_dim, 2*vis_input_dim),
            nn.Linear(2*vis_input_dim, embed_dim)
        )
        self.desc_encoder = nn.Linear(lang_input_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)


    def process_state_feat(self, state_feat):
        ''' Reduce the input state features

        This function combines the visual states between adjacent actions.

        Args:
            state_feat:         Visual observations of procedures.  (batch, time_horz, 2, vis_input_dim)

        Returns:
            state_feat_tmp:     Reduced visual observations.    (batch, time_horz+1, vis_input_dim)
        '''

        state_feat_tmp_1 = torch.cat([state_feat[:, :, 0:1, :], state_feat[:, -1:, -1:, :]], 1) # [batch, time_horz+1, 1, dim]
        state_feat_tmp_2 = torch.cat([state_feat[:, 0:1, 0:1, :], state_feat[:, :, -1:, :]], 1)
        state_feat_tmp = torch.cat([state_feat_tmp_1, state_feat_tmp_2], 2) # [batch, time_horz+1, 2, dim]
        state_feat_tmp = state_feat_tmp.mean(2) # [batch, time_horz+1, dim]

        return state_feat_tmp


    def forward(self, state_feat, all_state_desc_feat):
        ''' Forward pass of the state encoder

        This function first encodes the state feature and the description feature,
        then computes their similarity.

        Args
            state_feat:             Visual observations of procedures.      (batch, time_horz, 2, vis_input_dim)
            all_state_desc_feat:    Descriptions of before and after state of all actions.     (num_action, num_prompts, lang_input_dim)

        Returns:
            state_feat:             Encoded start and end visual features.    (batch, 2, embed_dim)
            inter_state_feat:       Encoded intermediate visual features.   (batch, time_horz-1, embed_dim)
            state_logits:           Similarity between visual and linguistic features.   (batch, 2, num_action)
            prompt_features:        Encoded description features.   (num_action, num_prompts, embed_dim)
        '''

        batch_size = state_feat.shape[0]

        state_feat = self.process_state_feat(state_feat) # [batch, time_horz+1, input_dim]

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
        s_logits = img_text_similarlity(start_state_feat, start_desc_feat, self.logit_scale.exp())
        e_logits = img_text_similarlity(end_state_feat, end_desc_feat, self.logit_scale.exp())
   
        s_logits = s_logits.reshape(batch_size, n_actions, -1).mean(-1)
        e_logits = e_logits.reshape(batch_size, n_actions, -1).mean(-1)
        state_logits = torch.cat((s_logits.unsqueeze(1), e_logits.unsqueeze(1)), 1) # [batch, 2, n_actions]
        state_feat = torch.cat([start_state_feat.unsqueeze(1), end_state_feat.unsqueeze(1)], 1)

        return state_feat, inter_state_feat, state_logits, prompt_features
