import torch.nn as nn
from .multiHeadAttention import * 
from .positionWiseFFN import *

class DecoderLayer(nn.Module):
    def __init__(self, n_head, model_dim, dropout):
        super().__init__()
        self.mha = nn.ModuleList([
            MultiHeadAttention(n_head, model_dim, dropout) for _ in range(2)
        ])
        self.ffn = PositionWiseFFN(model_dim, dropout)
    
    def forward(self, yz, xz, yz_look_ahead_mask, xz_pad_mask):
        # yz: (batch_size, yz_seq_len, model_dim)
        # xz: (batch_size, xz_seq_len, model_dim)
        # yz_look_ahead_mask: (batch_size, 1, yz_seq_len, yz_seq_len)
        # xz_pad_mask: (batch_size, 1, 1, xz_seq_len)
        dec_output = self.mha[0](yz, yz, yz, yz_look_ahead_mask)
        dec_output = self.mha[1](dec_output, xz, xz, xz_pad_mask)
        dec_output = self.ffn(dec_output)
        return dec_output

class Decoder(nn.Module):
    def __init__(self, n_head, model_dim, dropout, n_layer):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(n_head, model_dim, dropout) for _ in range(n_layer)])

    def forward(self, yz, xz, yz_look_ahead_mask, xz_pad_mask):
        # yz: (batch_size, yz_seq_len, model_dim)
        # xz: (batch_size, xz_seq_len, model_dim)
        # yz_look_ahead_mask: (batch_size, 1, yz_seq_len, yz_seq_len)
        # xz_pad_mask: (batch_size, 1, 1, xz_seq_len)
        for layer in self.layers:
            yz = layer(yz, xz, yz_look_ahead_mask, xz_pad_mask)
        return yz