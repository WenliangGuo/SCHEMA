import torch
import torch.nn as nn
import numpy as np
from models.modules import PositionalEmbedding, transformer_layer, TransFormerDecoder

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
            uncertainty = False,
            dataset = "crosstask"
        ):
        super().__init__()
        self.time_horz = time_horz
        self.embed_dim = embed_dim
        self.uncertainty = uncertainty
        self.dataset = dataset

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

        # if self.dataset == "crosstask":
        if self.dataset == "crosstask" or self.dataset == "coin" or self.dataset == "niv":
            if self.uncertainty is False:
                self.query_embed = nn.Linear(num_tasks, embed_dim)

    def process_state_query(self, state_feat, tasks):
        ''' Process the state query

        This function first generates mask tokens, and concatenates them with the start-end visual tokens
        to make the input query. Then it adds positional encoding to the query. The task information is
        also introduced into the query optionally.

        Args:
            state_feat:     Encoded start and end visual features.    (batch, time_horz+1, embed_dim)
            tasks:          Task. (batch)

        Returns:
            query:          Processed query. (time_horz+1, batch_size, embed_dim)
        '''

        batch, _, embed_dim = state_feat.shape

        if self.uncertainty is True:
            init_tensor = torch.randn([batch, 1, embed_dim]).to(state_feat.device)
            init_tensor = init_tensor.repeat(1, self.time_horz-1, 1)
        else:
            init_tensor = torch.zeros([batch, self.time_horz-1, embed_dim]).to(state_feat.device)
        init_tensor = torch.zeros([batch, self.time_horz-1, embed_dim]).to(state_feat.device)
        state_feat_tmp = torch.cat([state_feat[:, :1, :], init_tensor, state_feat[:, -1:, :]], 1)

        query = self.pos_encoder(state_feat_tmp) # [batch_size, time_horz+1, embed_dim]
        query = query.permute(1, 0, 2)

        # adding predicted task infomation to query for crosstask increases performance
        # if self.dataset == "crosstask":
        if self.dataset == "crosstask" or self.dataset == "coin" or self.dataset == "niv":
            if self.uncertainty is False:
                task_query = self.query_embed(tasks.clone().detach()).expand(self.time_horz + 1, -1, -1)
                query = query + task_query

        return query


    def forward(self, state_feat, state_prompt_features, tasks):
        ''' Forward pass of the state decoder

        State decoder takes the encoded state feature as query, and the encoded description 
        features as memory, then outputs the predicted state features.

        Args:
            state_feat:             Encoded start and end visual features.    (batch, 2, embed_dim)
            state_prompt_features:  Encoded description features.   (num_action, num_prompts, embed_dim)
            tasks:                  Task. (batch)

        Returns:
            state_feat:             Predicted visual features.    (batch, time_horz+1, embed_dim)
        '''

        batch_size, _, _ = state_feat.shape

        # Generate Query for Transformer
        state_query = self.feat_encode(state_feat)
        state_query = self.process_state_query(state_query, tasks)  # [time_horz+1, batch_size, embed_dim]

        # Generate Memory(Key and Value) for Transformer
        memory = state_prompt_features.reshape(-1, state_prompt_features.shape[-1])
        memory = memory.unsqueeze(1).repeat(1, batch_size, 1)

        state_output = self.decoder(state_query, memory, memory)
        
        state_output = state_output[:, 1:-1, :]
        state_output = self.state_proj(state_output)
        state_feat = torch.cat([state_feat[:, 0:1, :], state_output, state_feat[:, -1:, :]], 1)

        return state_feat
