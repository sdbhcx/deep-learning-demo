# 修复导入语句，添加必要的模块和函数
import pytorch.nn as nn
import pytorch.th as th
import numpy as np
import pytorch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 添加缺失的函数定义
relu = F.relu
def softmax(x, dim):
    return F.softmax(x, dim=dim)
def cross_entropy(input, target):
    return F.cross_entropy(input, target)

# 定义缺失的DateDataset类和pad_zero函数
class DateDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        # 这里简化实现，实际使用时需要根据具体任务定义
        self.num_word = 1000  # 假设词汇表大小为1000
        self.index2word = {i: str(i) for i in range(self.num_word)}  # 简单的索引到词的映射
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        # 这里简化实现，返回一些示例数据
        # 实际使用时需要根据具体任务生成真实数据
        input_data = np.random.randint(1, self.num_word, size=10)
        target_data = np.random.randint(1, self.num_word, size=11)  # 目标序列长度比输入长1
        return input_data, target_data, idx

def pad_zero(seqs, max_len):
    padded_seqs = []
    for seq in seqs:
        if len(seq) < max_len:
            # 前面补零还是后面补零取决于任务需求
            padded_seq = np.pad(seq, (0, max_len - len(seq)), 'constant')
        else:
            padded_seq = seq[:max_len]
        padded_seqs.append(padded_seq)
    return np.array(padded_seqs)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        # 每个注意力头的维度
        self.head_dim=model_dim//n_head
        # 注意力头的数量
        self.n_head=n_head
        # 模型的维度
        self.model_dim=model_dim
        # 初始化线性变换层，用于生成query、key和value
        self.wq=nn.Linear(model_dim, n_head * self.head_dim)
        self.wk=nn.Linear(model_dim, n_head * self.head_dim)
        self.wv=nn.Linear(model_dim, n_head * self.head_dim)
        # 输出的全连接层
        self.output_dense=nn.Linear(model_dim, model_dim)
        # Dropout层，用于防止模型过拟合
        self.output_drop=nn.Dropout(drop_rate)
        # 层标准化，用于稳定神经网络的训练
        self.layer_norm=nn.LayerNorm(model_dim)
        self.attention=None
    def forward(self, q, k, v, mask):
        # 保存原始输入q，用于后续的残差连接
        residual=q
        # 分别对输入的q、k、v做线性变换，生成query、key和value
        query=self.wq(q)
        key=self.wk(k)
        value=self.wv(v)
        # 对生成的query、key和value进行头分割，以便进行多头注意力计算
        query=self.split_heads(query)
        key=self.split_heads(key)
        value=self.split_heads(value)
        # 计算上下文向量
        context=self.scaled_dot_product_attention(query, key, value, mask)
        # 对上下文向量进行线性变换
        output=self.output_dense(context)
        # 添加dropout
        output=self.output_drop(output)
        # 添加残差连接并进行层标准化
        output=self.layer_norm(residual+output)
        return output
    def split_heads(self, x):
        # 将输入x的形状(shape)变为(n, step, n_head, head_dim)，然后重排，得到(n, n_head, step, head_dim)
        x=th.reshape(x, (x.shape[0], x.shape[1], self.n_head, self.head_dim))
        return x.permute(0, 2, 1, 3)
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # 计算缩放因子
        dk=th.tensor(k.shape[-1]).type(th.float)
        # 计算注意力分数
        score=th.matmul(q, k.permute(0, 1, 3, 2))/(th.sqrt(dk)+1e-8)
        if mask is not None:
            # 如果提供了mask，则将mask位置的分数设置为负无穷，使得这些位置的softmax值接近0
            score=score.masked_fill_(mask,-np.inf)
        # 应用softmax函数计算得到注意力权重
        self.attention=softmax(score,dim=-1)
        # 计算上下文向量
        context=th.matmul(self.attention,v)
        # 重排上下文向量的维度并进行维度合并
        context=context.permute(0, 2, 1, 3)
        context=context.reshape((context.shape[0], context.shape[1],-1))
        return context


class PositionWiseFFN(nn.Module):
    def __init__(self, model_dim, dropout=0.0):
        super().__init__()
        # 前馈神经网络的隐藏层维度，设为模型维度的4倍
        ffn_dim=model_dim * 4
        # 第一个线性变换层，其输出维度为前馈神经网络的隐藏层维度
        self.linear1=nn.Linear(model_dim, ffn_dim)
        # 第二个线性变换层，其输出维度为模型的维度
        self.linear2=nn.Linear(ffn_dim, model_dim)
        # Dropout层，用于防止模型过拟合
        self.dropout=nn.Dropout(dropout)
        # 层标准化，用于稳定神经网络的训练
        self.layer_norm=nn.LayerNorm(model_dim)
    def forward(self, x):
        # 对输入x进行前馈神经网络的计算
        # 首先，通过第一个线性变换层并使用ReLU作为激活函数
        output=relu(self.linear1(x))
        # 然后，通过第二个线性变换层
        output=self.linear2(output)
        # 接着，对上述输出进行dropout操作
        output=self.dropout(output)
        # 最后，对输入x和前馈神经网络的输出做残差连接，然后进行层标准化
        output=self.layer_norm(x+output)
        return output  # 返回结果，其形状为[n, step, dim]
    

class EncoderLayer(nn.Module):
    def __init__(self, n_head, emb_dim, drop_rate):
        super().__init__()
        # 多头注意力机制层
        self.mha=MultiHeadAttention(n_head, emb_dim, drop_rate)
        # 前馈神经网络层
        self.ffn=PositionWiseFFN(emb_dim, drop_rate)
    def forward(self, xz, mask):
        # xz的形状为 [n, step, emb_dim]
        # 通过多头注意力机制层处理xz，得到context，其形状也为 [n, step, emb_dim]
        context=self.mha(xz, xz, xz, mask)
        # 将context传入前馈神经网络层，得到输出
        output=self.ffn(context)
        return output
class Encoder(nn.Module):
    def __init__(self, n_head, emb_dim, drop_rate, n_layer):
        super().__init__()
        # 定义n_layer个EncoderLayer，保存在ModuleList中
        self.encoder_layers=nn.ModuleList(
            [EncoderLayer(n_head, emb_dim, drop_rate) for _ in range(n_layer)]
        )
    # 修复循环提前返回的问题
    def forward(self, xz, mask):
        # 依次通过所有的EncoderLayer
        for encoder in self.encoder_layers:
            xz=encoder(xz, mask)
        # 将return语句移到循环外，确保所有层都被执行
        return xz # 返回的xz形状为 [n, step, emb_dim]
    

class DecoderLayer(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        # 定义两个多头注意力机制层
        self.mha=nn.ModuleList([MultiHeadAttention(n_head, model_dim, drop_rate) for _ in range(2)])
        # 定义一个前馈神经网络层
        self.ffn=PositionWiseFFN(model_dim, drop_rate)
    def forward(self, yz, xz, yz_look_ahead_mask, xz_pad_mask):
        # 执行第一个注意力层的计算，3个输入均为yz，使用自注意力机制
        dec_output=self.mha[0](yz, yz, yz, yz_look_ahead_mask)  # [n, step, model_dim]
        # 执行第二个注意力层的计算，其中Q来自前一个注意力层的输出，K和V来自编码器的输出
        dec_output=self.mha[1](dec_output, xz, xz, xz_pad_mask)  # [n, step, model_dim]
        # 通过前馈神经网络层
        dec_output=self.ffn(dec_output)    # [n, step, model_dim]
        return dec_output
class Decoder(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate, n_layer):
        super().__init__()
        # 定义n_layer个DecoderLayer，保存在ModuleList中
        self.num_layers=n_layer
        self.decoder_layers=nn.ModuleList(
            [DecoderLayer(n_head, model_dim, drop_rate) for _ in range(n_layer)]
        )
    # 修复循环提前返回的问题
    def forward(self, yz, xz, yz_look_ahead_mask, xz_pad_mask):
        # 依次通过所有的DecoderLayer
        for decoder in self.decoder_layers:
            yz=decoder(yz, xz, yz_look_ahead_mask, xz_pad_mask)
        # 将return语句移到循环外，确保所有层都被执行
        return yz  # 返回的yz形状为 [n, step, model_dim]
        


class PositionEmbedding(nn.Module):
    def __init__(self, max_len, emb_dim, n_vocab):
        super().__init__()
        # 生成位置编码矩阵
        pos=np.expand_dims(np.arange(max_len), 1)  # [max_len, 1]
        # 使用正弦和余弦函数生成位置编码
        pe=pos/np.power(1000, 2*np.expand_dims(np.arange(emb_dim)//2, 0)/emb_dim)
        pe[:, 0::2]=np.sin(pe[:, 0::2])
        pe[:, 1::2]=np.cos(pe[:, 1::2])
        pe=np.expand_dims(pe, 0)  # [1, max_len, emb_dim]
        self.pe=th.from_numpy(pe).type(th.float32)
        # 定义词嵌入层
        self.embeddings=nn.Embedding(n_vocab, emb_dim)
        # 初始化词嵌入层的权重
        self.embeddings.weight.data.normal_(0, 0.1)
    def forward(self, x):
        # 确保位置编码在与词嵌入权重相同的设备上
        device=self.embeddings.weight.device
        self.pe=self.pe.to(device)
        # 计算输入的词嵌入权重，并加上位置编码
        x_embed=self.embeddings(x)+self.pe  # [n, step, emb_dim]
        return x_embed  # [n, step, emb_dim]
    

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

# 修复变量定义顺序问题
# 首先定义常量和创建数据集
MAX_LENGTH = 20  # 定义最大序列长度
# 创建一个数据集，包含1000个样本
dataset=DateDataset(1000)
# 现在可以使用dataset.num_word来初始化模型
# 初始化一个Transformer模型，设置词汇表大小、最大序列长度、层数、嵌入维度、多头注意力的头数、dropout比率和填充标记的索引
model=Transformer(n_vocab=dataset.num_word, max_len=MAX_LENGTH, n_layer=3, emb_dim=32, n_head=8, drop_rate=0.1, padding_idx=0)
# 检测是否有可用的GPU，如果有，则使用GPU进行计算；如果没有，则使用CPU
device=th.device("cuda" if th.cuda.is_available() else "cpu")
# 将模型移动到相应的设备（CPU或GPU）
model=model.to(device)
# 创建一个数据加载器，设定批量大小为32，每个批量的数据会被打乱
dataloader=DataLoader(dataset, batch_size=32, shuffle=True)
# 执行10个训练周期
for i in range(10):
    # 对于数据加载器中的每批数据，对输入和目标张量进行零填充，使其长度达到最大，然后将其转换为PyTorch张量，并移动到相应的设备（CPU或GPU）
    for input_tensor, target_tensor, _ in dataloader:
        input_tensor=th.from_numpy(
            pad_zero(input_tensor, max_len=MAX_LENGTH)).long().to(device)
        target_tensor=th.from_numpy(
            pad_zero(target_tensor, MAX_LENGTH+1)).long().to(device)
        # 使用模型的step方法进行一步训练，并获取损失值
        loss, _=model.step(input_tensor, target_tensor)
    # 打印每个训练周期后的损失值
    print(f"epoch: {i+1}, 	loss: {loss}")


def evaluate(model, x, y):
    model.eval()
    x=th.from_numpy(pad_zero([x], max_len=MAX_LENGTH)).long().to(device)
    y=th.from_numpy(pad_zero([y], max_len=MAX_LENGTH)).long().to(device)
    decoder_outputs=model(x, y)
    _, topi=decoder_outputs.topk(1)
    decoded_ids=topi.squeeze()
    decoded_words=[]
    for idx in decoded_ids:
        decoded_words.append(dataset.index2word[idx.item()])
    return ''.join(decoded_words)



class Encoder(nn.Module):
    def __init__(self, dim, layer_num=3):
         super(Encoder, self).__init__()
         self.convs=nn.ModuleList([DownConvLayer(dim) for _ in range(layer_num)])
    def forward(self, x):
         for conv in self.convs:
               x=conv(x)
         return x
class Decoder(nn.Module):
    def __init__(self, dim, layer_num=3):
         super(Decoder, self).__init__()
         self.convs=nn.ModuleList([UpConvLayer(dim) for _ in range(layer_num)])
         self.final_conv=nn.Conv2d(dim, 1, 3, stride=1, padding=1)
    def forward(self, x):
         for conv in self.convs:
               x=conv(x)
         reconstruct=self.final_conv(x)
         return reconstruct
    

class AutoEncoderModel(nn.Module):
    def __init__(self):
        super(AutoEncoderModel, self).__init__()
        self.encoder=Encoder(1, layer_num=1)
        self.decoder=Decoder(1, layer_num=1)
    def forward(self, inputs):
        latent=self.encoder(inputs)
        reconstruct_img=self.decoder(latent)
        return reconstruct_img
# 加载和预处理MNIST数据集
transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])
train_dataset=torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
# 创建自编码器模型实例、优化器和损失函数
model=AutoEncoderModel()
optimizer=th.optim.Adam(model.parameters(), lr=1e-2)
criterion=nn.MSELoss()
# 创建学习率调度器
scheduler=th.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# 训练自编码器模型
num_epochs=10
device=th.device("cuda" if th.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(num_epochs):
    running_loss=0.0
    for data in train_loader:
        images, _=data
        images=images.to(device)
        optimizer.zero_grad()
        reconstructed_images=model(images)
        loss=criterion(images, reconstructed_images)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    scheduler.step()
    epoch_loss=running_loss/len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
print("Training finished!")


class VAEModel(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAEModel, self).__init__()
        self.encoder=nn.Linear(784, 400)
        self.fc_mu=nn.Linear(400, latent_dim)
        self.fc_var=nn.Linear(400, latent_dim)
        self.fc_decode=nn.Linear(latent_dim, 400)
        self.decoder=nn.Linear(400, 784)
    def encode(self, x):
        x=F.relu(self.encoder(x))
        return self.fc_mu(x), self.fc_var(x)
    def reparameterize(self, mu, log_var):
        std=0.5 * th.exp(log_var)
        eps=th.randn_like(std)
        return mu+eps * std
    def decode(self, z):
        x=F.relu(self.fc_decode(z))
        return th.sigmoid(self.decoder(x))
    def forward(self, x):
        mu, log_var=self.encode(x)
        z=self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
def vae_loss(reconstructed, original, mu, log_var):
    recon_loss=F.binary_cross_entropy(reconstructed, original, reduction="sum")
    kl_divergence=-0.5 * th.sum(1+log_var-mu.pow(2)-log_var.exp())
    return recon_loss+kl_divergence