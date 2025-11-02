import torch.nn as nn
import numpy as np
from .multiHeadAttention import * 
from .positionWiseFFN import *
from .encoder import *
from .decoder import *

class PositionEmbedding(nn.Module):
    def __init__(self, max_len, emb_dim, n_vocab):
        super().__init__()
        pos = np.expand_dims(np.arange(max_len), 1) #[max_len, 1]
        pe = pos/np.power(1000, 2*np.expand_dims(np.arange(emb_dim) // 2, 0)/emb_dim)
        pe[:,0::2] = np.sin(pe[:,0::2])
        pe[:,1::2] = np.cos(pe[:,1::2])
        pe = np.expand_dims(pe, 0) #[1, max_len, emb_dim]
        self.pe = th.from_numpy(pe).type(th.float32)
        self.embeddings = nn.Embedding(n_vocab, emb_dim)
        self.embeddings.weight.data.normal_(0, 0.1)
    
    def forward(self, x):
        x_embed = self.embeddings(x) + self.pe
        return x_embed

class Transformer(nn.Module):
    def __init__(self, n_head, model_dim, dropout, n_layer, max_len, emb_dim, n_vocab):
        super().__init__()
        self.encoder = Encoder(n_head, model_dim, dropout, n_layer, max_len, emb_dim, n_vocab)
        self.decoder = Decoder(n_head, model_dim, dropout, n_layer, max_len, emb_dim, n_vocab)
        self.position_embedding = PositionEmbedding(max_len, emb_dim, n_vocab)