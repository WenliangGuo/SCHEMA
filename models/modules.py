import torch
import torch.nn as nn
import numpy as np
import copy
import math
from torch.nn import MultiheadAttention, LayerNorm

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Attention(nn.Module):
    def __init__(self, hidden_size, att_heads, dropout):
  
        super().__init__()
        self.attention = MultiheadAttention(hidden_size, att_heads)
        self.dropout = nn.Dropout(dropout)
        self.ln = LayerNorm(hidden_size)

    def forward(self, query, key, value):
        '''
        input: [batch_size, seq_len, hidden_size]
        '''
        attn_output, _ = self.attention(query, key, value)
        attn_output = self.dropout(attn_output)
        attn_output = self.ln(attn_output)
        return attn_output


class transformer_layer(nn.Module):
    def __init__(self, hidden_size, att_heads, dropout, mlp_ratio):

        super().__init__()
        self.self_attn = Attention(hidden_size=hidden_size, att_heads=att_heads, dropout=dropout)
        self.cross_attn = Attention(hidden_size=hidden_size, att_heads=att_heads, dropout=dropout)
        self.norm1 = LayerNorm(hidden_size)
        self.norm2 = LayerNorm(hidden_size)
        self.norm3 = LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * mlp_ratio, hidden_size)
        )

    def forward(self, query, key, value):
        '''
        x: [batch_size, seq_len, hidden_size]
        '''

        batch_size, _, _ = query.shape

        out = self.self_attn(query, query, query)
        query = query + self.dropout1(out)
        query = self.norm1(query)

        out = self.cross_attn(query, key, value)
        query = query + self.dropout2(out)
        query = self.norm2(query)
        
        out = self.mlp(query)
        query = query + self.dropout3(out)
        query = self.norm3(query)

        return query


class TransFormerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, embed_dim, norm=None):
        """TransformerDecoder is a stack of N decoder layers

        Args:
            decoder_layer: an instance of the TransformerDecoderLayer() class (required).
            num_layers: the number of sub-decoder-layers in the decoder (required).
            norm: the layer normalization component (optional).

        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, key, value):
        output = query

        for i, layer in enumerate(self.layers):
            output = layer(output, key, value)

        if self.norm is not None:
            output = self.norm(output)
        output = output.permute(1, 0, 2)

        return output


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe