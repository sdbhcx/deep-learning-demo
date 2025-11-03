import torch.nn as nn
import numpy as np
from .multiHeadAttention import * 
from .positionWiseFFN import *
from .encoder import *
from .decoder import *

"""
位置编码层， 将序列中的每个位置编码为一个固定大小的向量， 向量的维度为emb_dim
"""
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
    def __init__(self, n_vocab, max_len, n_layer=6, emb_dim=512, n_head=8, drop_rate=0.1, padding_idx=0):
        super().__init__()
        # 初始化最大长度、填充索引、词汇表大小
        self.max_len=max_len
        self.padding_idx=th.tensor(padding_idx)
        self.dec_v_emb=n_vocab
        # 初始化位置嵌入、编码器、解码器和输出层
        self.embed=PositionEmbedding(max_len, emb_dim, n_vocab)
        self.encoder=Encoder(n_head, emb_dim, drop_rate, n_layer)
        self.decoder=Decoder(n_head, emb_dim, drop_rate, n_layer)
        self.output=nn.Linear(emb_dim, n_vocab)
        # 初始化优化器
        self.opt=th.optim.Adam(self.parameters(), lr=0.002)
    def forward(self, x, y):
        # 对输入和目标进行嵌入
        x_embed, y_embed=self.embed(x), self.embed(y)
        # 创建填充掩码
        pad_mask=self._pad_mask(x)
        # 对输入进行编码
        encoded_z=self.encoder(x_embed, pad_mask)
        # 创建前瞻掩码
        yz_look_ahead_mask=self._look_ahead_mask(y)
        # 将编码后的输入和前瞻掩码传入解码器
        decoded_z=self.decoder(
            y_embed, encoded_z, yz_look_ahead_mask, pad_mask)
        # 通过输出层得到最终输出
        output=self.output(decoded_z)
        return output
    def step(self, x, y):
        # 清空梯度
        self.opt.zero_grad()
        # 计算输出和损失
        logits=self(x, y[:, :-1])
        loss=cross_entropy(logits.reshape(-1, self.dec_v_emb), y[:, 1:].reshape(-1))
        # 进行反向传播
        loss.backward()
        # 更新参数
        self.opt.step()
        return loss.cpu().data.numpy(), logits
    def _pad_bool(self, seqs):
        # 创建掩码，标记哪些位置是填充的
        return th.eq(seqs, self.padding_idx)
    def _pad_mask(self, seqs):
        # 将填充掩码扩展到合适的维度
        len_q=seqs.size(1)
        mask=self._pad_bool(seqs).unsqueeze(1).expand(-1, len_q,-1)
        return mask.unsqueeze(1)
    def _look_ahead_mask(self, seqs):
        # 创建前瞻掩码，防止在生成序列时看到未来位置的信息
        device=next(self.parameters()).device
        _, seq_len=seqs.shape
        mask=th.triu(th.ones((seq_len, seq_len), dtype=th.long),
                    diagonal=1).to(device)
        mask=th.where(self._pad_bool(seqs)[:, None, None, :], 1, mask[None, None, :, :]).to(device)
        return mask>0