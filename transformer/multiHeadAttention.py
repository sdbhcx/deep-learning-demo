import torch.nn as nn
import torch as th
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate) -> None:
        super().__init__()
        # 每个头的维度
        self.head_dim = model_dim // n_head
        # 注意力头的维度
        self.n_head = n_head
        # 模型的维度
        self.model_dim=model_dim
        # 初始化线性变换层，用于生成query、key和value
        self.wq = nn.Linear(model_dim, n_head * self.head_dim)
        self.wk = nn.Linear(model_dim, n_head * self.head_dim)
        self.wv = nn.Linear(model_dim, n_head * self.head_dim)
        # 输出的全连接层
        self.output_dense = nn.Linear(model_dim, model_dim)
        # Dropout层，用于防止模型过拟合
        self.output_dense = nn.Dropout(drop_rate)
        # 层标准化，用于稳定神经网络的训练
        self.layer_norm = nn.LayerNorm(model_dim)
        self.attention=None
    def forward(self, q, k, v, mask):
        residual = q
        query = self.wq(q)
        key = self.wk(k)
        value = self.wv(v)
        # 对生成的query、key和value进行头分割，以便进行多头注意力计算
        query=self.split_heads(query)
        key=self.split_heads(key)
        value=self.split_heads(value)
        # 计算上下文向量
        context = self.scaled_dot_product_attention(query, key, value, mask)
        # 对上下文向量进行线性变换
        output = self.output_dense(context)
        # 添加dropout
        output = self.output_drop(output)
        # 添加残差连接并进行层标准化
        output = self.layer_norm(residual + output)
        return output
    
    def split_heads(self, x):
        # 将输入x的形状(shape)变为(n, step, n_head, head_dim)，然后重排，得到(n, n_head, step, head_dim)
        x = th.reshape(x, (x.shape[0], x.shape[1], self.n_head, self.head_dim))
        return x.permute(0, 2, 1, 3)
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # 计算缩放因子
        dk = th.tensor(k.shape[-1]).type(th.float)
        # 计算注意力分数
        scores = th.matmul(q, k.permute(0,1,3,2))/th.sqrt(dk)
        if mask is not None:
            score = score.masked_fill_(mask, -np.inf)
        self.attention = th.softmax(score, dim=-1)
        # 计算上下文向量
        context=th.matmul(self.attention,v)
        # 对上下文向量进行头合并
        context=context.permute(0,2,1,3).contiguous()
        # 对上下文向量进行线性变换
        context=th.reshape(context, (context.shape[0], context.shape[1], self.model_dim))
        return context