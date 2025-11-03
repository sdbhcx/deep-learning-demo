import torch as th
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 简化的多头注意力机制实现，用于演示其作用
class SimplifiedMultiHeadAttention(nn.Module):
    def __init__(self, model_dim, n_head):
        super().__init__()
        self.model_dim = model_dim  # 模型维度
        self.n_head = n_head        # 注意力头数量
        self.head_dim = model_dim // n_head  # 每个头的维度
        
        # 线性变换层
        self.wq = nn.Linear(model_dim, model_dim)
        self.wk = nn.Linear(model_dim, model_dim)
        self.wv = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        
    def split_heads(self, x):
        # 输入: [batch_size, seq_len, model_dim]
        batch_size, seq_len, _ = x.size()
        # 转换为: [batch_size, seq_len, n_head, head_dim]
        x = x.view(batch_size, seq_len, self.n_head, self.head_dim)
        # 重排为: [batch_size, n_head, seq_len, head_dim]
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v):
        # 线性变换
        query = self.wq(q)
        key = self.wk(k)
        value = self.wv(v)
        
        # 分割多头
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        
        # 缩放点积注意力
        d_k = query.size(-1)
        scores = th.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        attn_weights = th.softmax(scores, dim=-1)
        context = th.matmul(attn_weights, value)
        
        # 合并多头
        batch_size = context.size(0)
        seq_len = context.size(2)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, self.model_dim)
        
        # 最终线性变换
        output = self.out_proj(context)
        
        return output, attn_weights

# 比较单头和多头注意力的效果
def compare_attention_heads():
    # 设置随机种子以确保结果可复现
    th.manual_seed(42)
    
    # 参数设置
    batch_size = 1
    seq_len = 5
    model_dim = 64
    
    # 创建输入数据
    x = th.randn(batch_size, seq_len, model_dim)
    
    # 创建不同头数的注意力机制
    attention_1head = SimplifiedMultiHeadAttention(model_dim, n_head=1)
    attention_8head = SimplifiedMultiHeadAttention(model_dim, n_head=8)
    
    # 前向传播
    _, attn_1head = attention_1head(x, x, x)
    _, attn_8head = attention_8head(x, x, x)
    
    # 可视化注意力权重
    visualize_attention_comparison(attn_1head, attn_8head)

# 可视化单头和多头注意力权重对比
def visualize_attention_comparison(attn_1head, attn_8head):
    plt.figure(figsize=(15, 10))
    
    # 绘制单头注意力
    plt.subplot(2, 1, 1)
    plt.imshow(attn_1head[0, 0].detach().numpy(), cmap='viridis')
    plt.colorbar(label='注意力权重')
    plt.title('单头注意力机制 (n_head=1)')
    plt.xlabel('键位置')
    plt.ylabel('查询位置')
    
    # 绘制多头注意力的前4个头
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for i in range(4):
        axes[i].imshow(attn_8head[0, i].detach().numpy(), cmap='viridis')
        axes[i].set_title(f'多头注意力 - 头 {i+1}')
        axes[i].set_xlabel('键位置')
        axes[i].set_ylabel('查询位置')
        plt.colorbar(axes[i].imshow(attn_8head[0, i].detach().numpy(), cmap='viridis'), ax=axes[i])
    
    plt.tight_layout()
    plt.savefig('multihead_attention_comparison.png')
    print("注意力权重可视化已保存为 multihead_attention_comparison.png")

# 演示多头注意力机制的关键特性
def demonstrate_key_features():
    print("=== 多头注意力机制的关键特性 ===")
    
    print("\n1. 多子空间表示学习:")
    print("   - 多头注意力将输入投影到多个不同的表示子空间")
    print("   - 每个头可以学习捕获序列中不同类型的关系和模式")
    print("   - 允许模型同时关注不同位置的不同语义信息")
    
    print("\n2. 表达能力提升:")
    print("   - 多头结构显著增加了模型的表达能力")
    print("   - 不同头学习到的注意力权重通常具有不同的模式")
    print("   - 模型可以同时关注短期和长期依赖关系")
    
    print("\n3. 计算效率:")
    print("   - 虽然参数数量略有增加，但计算复杂度仍然是O(n²)")
    print("   - 相比增加模型维度，多头设计在参数效率上更优")
    print("   - 支持并行计算，提高训练和推理效率")
    
    print("\n4. 应用场景:")
    print("   - 机器翻译: 捕捉源语言和目标语言之间的复杂对应关系")
    print("   - 文本摘要: 识别关键信息并生成简洁摘要")
    print("   - 问答系统: 关联问题和上下文信息")
    print("   - 图像描述生成: 连接视觉特征和文本生成")

# 模拟注意力头捕获不同关系
def demonstrate_relationship_capture():
    print("\n=== 多头注意力捕获不同关系的示例 ===")
    
    # 创建一个简单的序列，包含明显的模式
    # 假设序列中的元素遵循特定规则
    batch_size = 1
    seq_len = 10
    model_dim = 32
    
    # 创建一个带有模式的序列
    x = th.zeros(batch_size, seq_len, model_dim)
    
    # 为每个位置分配不同的特征模式
    for i in range(seq_len):
        x[0, i, :] = th.sin(th.tensor(i * 0.5)) + th.randn(model_dim) * 0.1
    
    # 创建多头注意力机制
    attention = SimplifiedMultiHeadAttention(model_dim, n_head=4)
    
    # 前向传播
    _, attn_weights = attention(x, x, x)
    
    # 分析不同头的注意力模式
    print(f"\n各头的注意力权重统计:")
    for head_idx in range(4):
        head_attn = attn_weights[0, head_idx].detach().numpy()
        # 计算平均注意力距离（关注近邻还是远邻）
        avg_dist = np.sum([np.sum([head_attn[i, j] * np.abs(i-j) for j in range(seq_len)]) for i in range(seq_len)])
        print(f"头 {head_idx+1}: 平均注意力距离 = {avg_dist:.2f}")
        
        # 找出每个位置最关注的位置
        max_attn_positions = np.argmax(head_attn, axis=1)
        print(f"   最关注位置: {max_attn_positions}")

if __name__ == "__main__":
    print("=== 多头注意力机制演示 ===")
    
    # 比较单头和多头注意力
    print("\n比较单头和多头注意力的效果:")
    compare_attention_heads()
    
    # 演示关键特性
    demonstrate_key_features()
    
    # 演示关系捕获
    demonstrate_relationship_capture()
    
    print("\n=== 总结 ===")
    print("多头注意力机制是Transformer架构的核心创新之一，通过将注意力机制扩展到多个头，")
    print("显著提高了模型捕获复杂模式和长距离依赖的能力，同时保持了计算效率。")