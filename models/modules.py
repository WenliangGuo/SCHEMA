import torch
import torch.nn as nn
from torch.nn import MultiheadAttention, LayerNorm
import numpy as np
import copy
import math

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

        # x = x.permute(1, 0, 2)
        # x = x + self.att(self.ln_1(x))
        # x = x + self.mlp(self.ln_2(x))

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
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512) [seq, batch, dim]
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """


    def __init__(self, decoder_layer, num_layers, embed_dim, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, key, value):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:

        Shape:
            see the docs in Transformer class.
        """
        output = query

        for i, layer in enumerate(self.layers):
            output = layer(output, key, value)

        if self.norm is not None:
            output = self.norm(output)
        output = output.permute(1, 0, 2)

        return output

""" Position Embedding """
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


""" Positional Encoding """
class PositionalAgentEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_t_len=200, max_a_len=200, concat=False, use_agent_enc=False, agent_enc_learn=False):
        super(PositionalAgentEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.d_model = d_model
        self.use_agent_enc = use_agent_enc
        if concat:
            self.fc = nn.Linear((3 if use_agent_enc else 2) * d_model, d_model)

        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)
        if use_agent_enc:
            if agent_enc_learn:
                self.ae = nn.Parameter(torch.randn(max_a_len, 1, d_model) * 0.1)
            else:
                ae = self.build_pos_enc(max_a_len)
                self.register_buffer('ae', ae)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def build_agent_enc(self, max_len):
        ae = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        ae[:, 0::2] = torch.sin(position * div_term)
        ae[:, 1::2] = torch.cos(position * div_term)
        ae = ae.unsqueeze(0).transpose(0, 1)
        return ae
    
    def get_pos_enc(self, num_t, num_a, t_offset):
        pe = self.pe[t_offset: num_t + t_offset, :]
        pe = pe.repeat_interleave(num_a, dim=0)
        return pe

    def get_agent_enc(self, num_t, num_a, a_offset, agent_enc_shuffle):
        if agent_enc_shuffle is None:
            ae = self.ae[a_offset: num_a + a_offset, :]
        else:
            ae = self.ae[agent_enc_shuffle]
        ae = ae.repeat(num_t, 1, 1)
        return ae

    def forward(self, x, num_a, agent_enc_shuffle=None, t_offset=0, a_offset=0):
        num_t = x.shape[0] // num_a
        pos_enc = self.get_pos_enc(num_t, num_a, t_offset)
        if self.use_agent_enc:
            agent_enc = self.get_agent_enc(num_t, num_a, a_offset, agent_enc_shuffle)
        if self.concat:
            feat = [x, pos_enc.repeat(1, x.size(1), 1)]
            if self.use_agent_enc:
                feat.append(agent_enc.repeat(1, x.size(1), 1))
            x = torch.cat(feat, dim=-1)
            x = self.fc(x)
        else:
            x = x + pos_enc
            if self.use_agent_enc:
                x = x + agent_enc
        return self.dropout(x)