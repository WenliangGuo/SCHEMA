import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import numpy as np
from models.modules import PositionalEmbedding, transformer_layer, TransFormerDecoder
from models.utils import img_text_similarlity

class ActionDecoder(nn.Module):
    def __init__(
            self, 
            embed_dim, 
            time_horz = 3, 
            att_heads = 8,
            mlp_ratio = 2,
            num_layers = 2,
            dropout = 0.1, 
            num_classes = 133,
            img_input_dim = 768,
            num_tasks = 18,
            init_weights = False,
            **kwargs
        ):
        super().__init__()
        self.time_horz = time_horz
        self.embed_dim = embed_dim
        
        # kwargs
        use_task = kwargs["use_task"] if "use_task" in kwargs else True
        use_state_pred = kwargs["use_state_pred"] if "use_state_pred" in kwargs else True
        use_state_memory = kwargs["use_state_memory"] if "use_state_memory" in kwargs else True
        use_action_proj_loss = kwargs["use_action_proj_loss"] if "use_action_proj_loss" in kwargs else True
        self.uncertainty = kwargs["uncertainty"] if "uncertainty" in kwargs else False

        self.use_task = use_task
        self.use_state_pred = use_state_pred
        self.use_state_memory = use_state_memory
        self.use_action_proj_loss = use_action_proj_loss

        self.use_random = kwargs["use_random"] if "use_random" in kwargs else False
        self.use_state_decoder = kwargs["use_state_decoder"] if "use_state_decoder" in kwargs else True

        ## Positional encoding
        if self.use_state_pred or not self.use_state_decoder:
            self.pos_encoder = PositionalEmbedding(
                d_model = embed_dim,
                max_len = 2 * self.time_horz + 1
            )
        else:
            self.pos_encoder = PositionalEmbedding(
                d_model = embed_dim,
                max_len = self.time_horz + 2
            )

        if self.use_task:
            self.query_embed = torch.nn.Embedding(num_tasks, embed_dim)

        # Transformer
        decoder_layer = transformer_layer(embed_dim, att_heads, dropout, mlp_ratio)
        self.decoder = TransFormerDecoder(decoder_layer, num_layers, embed_dim)
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        # Projection Layer
        if use_action_proj_loss:
            self.proj = nn.Linear(embed_dim, embed_dim)
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if not use_state_memory:
            # Memory
            self.memory = torch.nn.Embedding(img_input_dim, embed_dim)

        if not self.use_state_decoder:
            # Projection Layer
            self.state_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.state_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def process_state_query(self, state_feat, tasks):
        # [batch_size, time_horz+1, embed_dim]
        batch_size, num_steps, embed_dim = state_feat.shape

        if self.use_state_decoder:
            if self.use_random:
                init_tensor = torch.randn([batch_size, self.time_horz, embed_dim]).to(state_feat.device)
            elif self.uncertainty:
                init_tensor = torch.randn([batch_size, 1, embed_dim]).to(state_feat.device)
                init_tensor = init_tensor.repeat(1, self.time_horz, 1)
            else:
                init_tensor = torch.zeros([batch_size, self.time_horz, embed_dim]).to(state_feat.device)
            
            if self.use_state_pred:
                state_feat_tmp = torch.cat([torch.cat([state_feat[:, i:i+1], init_tensor[:, i:i+1]], dim=1) for i in range(num_steps-1)], dim=1)
                state_feat_tmp = torch.cat([state_feat_tmp, state_feat[:, -1:]], dim=1)
            else:
                state_feat_tmp = torch.cat([state_feat[:, :1], init_tensor, state_feat[:, -1:]], dim=1)
        else:
            init_tensor = torch.zeros([batch_size, self.time_horz * 2 - 1, embed_dim]).to(state_feat.device)
            state_feat_tmp = torch.cat([state_feat[:, :1], init_tensor, state_feat[:, -1:]], dim=1)

        query = self.pos_encoder(state_feat_tmp)
        query = query.permute(1, 0, 2)

        if self.use_task:
            task_query = self.query_embed(tasks).expand(query.shape[0], -1, -1)
            query = query + task_query 

        return query


    def forward(self, state_feat, prompt_features, tasks):
        '''
        input:
        state_feat:         (batch, time_horz+1, embed_dim)
        prompt_features:    (num_action, num_desc, embed_dim)
        tasks:              (batch)

        output:
        actoin_logits:      (batch, 2*num_action+1, num_action)
        state_feat:         (batch, 2*num_action+1, embed_dim)
        '''
        batch_size, _, _ = state_feat.shape

        ## get state positional encoding
        state_query = self.process_state_query(state_feat, tasks) # [time_horz+1, batch_size, embed_dim]

        if self.use_state_memory:
            # Option 1: Descriptions for cross-modal attention
            memory = prompt_features.reshape(-1, prompt_features.shape[-1]) #.clone().detach()
        else:
            # Option 2: Learnable Memory for cross-modal attention
            memory = self.memory.weight
        memory = memory.unsqueeze(1).repeat(1, batch_size, 1)
        # # Option 3: Action Descriptions for cross-modal attention
        # memory = action_prompt_features.reshape(-1, action_prompt_features.shape[-1])
        # memory = memory.unsqueeze(1).repeat(1, batch_size, 1)

        state_output = self.decoder(state_query, memory, memory)

        # action classification logits
        action_logits = self.classifier(state_output)

        if not self.use_state_decoder:
            state_pred = self.state_proj(state_output[:, 2:-1:2, :])
            state_feat = torch.cat([state_feat[:, :1, :], state_pred, state_feat[:, -1:, :]], 1)
            return state_feat, action_logits, state_output
      
        return action_logits, state_output
