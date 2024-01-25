import torch
import torch.nn as nn
from torch.nn import LayerNorm
import numpy as np
from models.modules import PositionalEmbedding, transformer_layer, TransFormerDecoder
from models.utils import img_text_similarlity

class StateDecoder(nn.Module):
    def __init__(
            self, 
            embed_dim, 
            time_horz = 3, 
            att_heads = 8,
            mlp_ratio = 2,
            num_layers = 2,
            dropout = 0.1, 
            img_input_dim = 768,
            num_classes = 133,
            num_tasks = 18,
            **kwargs
        ):
        super().__init__()
        self.time_horz = time_horz
        self.embed_dim = embed_dim
        self.use_random = kwargs["use_random"] if "use_random" in kwargs else False

        self.use_state_memory = kwargs["use_state_memory"] if "use_state_memory" in kwargs else True
        self.uncertainty = kwargs["uncertainty"] if "uncertainty" in kwargs else False
        self.dataset = kwargs["dataset"] if "dataset" in kwargs else None

        self.feat_encode = nn.Linear(embed_dim, embed_dim, bias=True)

        ## Positional encoding
        self.pos_encoder = PositionalEmbedding(
            d_model = embed_dim,
            max_len = self.time_horz + 1
        )

        # Transformer
        self.decoder = TransFormerDecoder(
            transformer_layer(
                embed_dim, 
                att_heads, 
                dropout, 
                mlp_ratio
            ), 
            num_layers, 
            embed_dim
        )

        # Projection Layer
        self.state_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if not self.use_state_memory and not self.use_observ_memory:
            self.memory = torch.nn.Embedding(img_input_dim, embed_dim)

        if self.dataset == "crosstask" and self.uncertainty is False:
            self.query_embed = nn.Linear(num_tasks, embed_dim)

    def process_state_query(self, state_feat, tasks):
        # [batch_size, time_horz+1, embed_dim]
        batch, _, embed_dim = state_feat.shape

        if self.use_random:
            init_tensor = torch.randn([batch, self.time_horz-1, embed_dim]).to(state_feat.device)
        elif self.uncertainty:
            init_tensor = torch.randn([batch, 1, embed_dim]).to(state_feat.device)
            init_tensor = init_tensor.repeat(1, self.time_horz-1, 1)
        else:
            init_tensor = torch.zeros([batch, self.time_horz-1, embed_dim]).to(state_feat.device)
        init_tensor = torch.zeros([batch, self.time_horz-1, embed_dim]).to(state_feat.device)
        state_feat_tmp = torch.cat([state_feat[:, :1, :], init_tensor, state_feat[:, -1:, :]], 1)

        query = self.pos_encoder(state_feat_tmp) # [batch_size, time_horz+1, embed_dim]
        query = query.permute(1, 0, 2)

        # adding predicted task infomation to query for crosstask increases performance
        if self.dataset == "crosstask" and self.uncertainty is False:
            task_query = self.query_embed(tasks.clone().detach()).expand(self.time_horz + 1, -1, -1)
            query = query + task_query

        return query


    def forward(self, state_feat, state_prompt_features, tasks):
        '''
        input:
        state_feat:             (batch, 2, embed_dim)
        state_prompt_features:  (batch, num_desc, embed_dim)
        tasks:                  (batch)

        output:
        state_feat:             (batch, time_horz+1, embed_dim)
        '''
        batch_size, _, _ = state_feat.shape

        ## get state positional encoding
        state_query = self.feat_encode(state_feat)
        state_query = self.process_state_query(state_query, tasks) # [time_horz+1, batch_size, embed_dim]

        if self.use_state_memory:
            # Option 1: Descriptions for cross-modal attention
            memory = state_prompt_features.reshape(-1, state_prompt_features.shape[-1])
        else:
            # Option 2: Learnable Memory for cross-modal attention
            memory = self.memory.weight
        memory = memory.unsqueeze(1).repeat(1, batch_size, 1)

        state_output = self.decoder(state_query, memory, memory)
        
        # index of state
        state_output = state_output[:, 1:-1, :]
        state_output = self.state_proj(state_output)
        state_feat = torch.cat([state_feat[:, 0:1, :], state_output, state_feat[:, -1:, :]], 1)

        return state_feat
