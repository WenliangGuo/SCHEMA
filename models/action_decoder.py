import torch
import torch.nn as nn
import numpy as np
from models.modules import PositionalEmbedding, transformer_layer, TransFormerDecoder

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
            uncertainty = False
        ):
        super().__init__()
        self.time_horz = time_horz
        self.embed_dim = embed_dim
        self.uncertainty = uncertainty

        ## Positional encoding
        self.pos_encoder = PositionalEmbedding(
            d_model = embed_dim,
            max_len = 2 * self.time_horz + 1
        )
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
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def process_state_query(self, state_feat, tasks):
        '''Process the state query
        
        This function first generates mask tokens, and concatenates them with the visual tokens
        to make the input query. Then it adds positional encoding to the query. The task information is
        then introduced into the query.

        Args:
            state_feat:     Predicted visual features.  (batch, time_horz+1, embed_dim)
            tasks:          Task.   (batch)

        Returns:
            query:          Processed state query.  (2*time_horz+1, batch, embed_dim)
        '''

        batch_size, num_steps, embed_dim = state_feat.shape

        if self.uncertainty:
            init_tensor = torch.randn([batch_size, 1, embed_dim]).to(state_feat.device)
            init_tensor = init_tensor.repeat(1, self.time_horz, 1)
        else:
            init_tensor = torch.zeros([batch_size, self.time_horz, embed_dim]).to(state_feat.device)
        
        state_feat_tmp = torch.cat([torch.cat([state_feat[:, i:i+1], init_tensor[:, i:i+1]], dim=1) for i in range(num_steps-1)], dim=1)
        state_feat_tmp = torch.cat([state_feat_tmp, state_feat[:, -1:]], dim=1)

        query = self.pos_encoder(state_feat_tmp)
        query = query.permute(1, 0, 2)
        task_query = self.query_embed(tasks).expand(query.shape[0], -1, -1)
        query = query + task_query 

        return query


    def forward(self, state_feat, state_prompt_features, tasks):
        '''Forward pass of the action decoder

        This function first generates the state query, and then uses the prompt features as memory to
        output the action logits.

        Args:
            state_feat:             Predicted visual features.  (batch, time_horz+1, embed_dim)
            state_prompt_features:  Encoded description features.   (num_action, num_desc, embed_dim)
            tasks:                  Task.   (batch)

        Returns:
            action_logits:          Predicted action logits.   (batch, time_horz, num_action)
        '''

        batch_size, _, _ = state_feat.shape

        # Generate Query for Transformer
        state_query = self.process_state_query(state_feat, tasks) # [time_horz+1, batch_size, embed_dim]

        # Generate Memory(Key and Value) for Transformer
        memory = state_prompt_features.reshape(-1, state_prompt_features.shape[-1])
        memory = memory.unsqueeze(1).repeat(1, batch_size, 1)

        state_output = self.decoder(state_query, memory, memory)
        state_action_logits = self.classifier(state_output)

        # Select action tokens
        action_idx = list(range(1, self.time_horz*2, 2))    # index of action/state
        action_logits = state_action_logits[:, action_idx, :]
      
        return action_logits
