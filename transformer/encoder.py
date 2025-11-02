import torch.nn as nn
from .multiHeadAttention import * 
from .positionWiseFFN import *

class EncoderLayer(nn.Module):
    def __init__(self, n_head, emb_dim, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(n_head, emb_dim, dropout)
        self.ffn = PositionWiseFFN(emb_dim, dropout)

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, emb_dim)
        x = self.mha(x, x, x, mask)
        x = self.ffn(x)
        return x

class Encoder(nn.Module):
    def __init__(self, n_head, emb_dim, dropout, n_layer):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(n_head, emb_dim, dropout) for _ in range(n_layer)])
    
    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, emb_dim)
        for layer in self.layers:
            x = layer(x, mask)
        return x