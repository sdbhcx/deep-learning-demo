import torch as th
import numpy as np
import matplotlib.pyplot as plt

# 创建一个简化版的MultiHeadAttention类来演示pad_mask的作用
class SimplifiedMultiHeadAttention:
    def __init__(self, n_head=1, model_dim=4, head_dim=4):
        self.n_head = n_head
        self.model_dim = model_dim
        self.head_dim = head_dim
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """简化的缩放点积注意力计算"""
        # 计算注意力分数
        scores = th.matmul(q, k.transpose(-2, -1)) / th.sqrt(th.tensor(self.head_dim, dtype=th.float))
        
        # 可视化原始注意力分数
        print("原始注意力分数:")
        print(scores.squeeze().detach().numpy())
        
        # 应用掩码（如果提供）
        if mask is not None:
            print("\n掩码:")
            print(mask.squeeze().detach().numpy())
            scores = scores.masked_fill(mask, -1e9)  # 填充为负无穷
            
            print("\n应用掩码后的注意力分数:")
            print(scores.squeeze().detach().numpy())
        
        # 计算注意力权重（softmax）
        attention_weights = th.softmax(scores, dim=-1)
        print("\n注意力权重（softmax后）:")
        print(attention_weights.squeeze().detach().numpy())
        
        # 计算上下文向量
        context = th.matmul(attention_weights, v)
        return context, attention_weights

# 演示pad_mask的作用
def demonstrate_pad_mask():
    print("=== Transformer中的pad_mask作用演示 ===")
    
    # 1. 创建模拟数据
    # 假设我们有一个批次大小为1，序列长度为4的输入
    batch_size = 1
    seq_len = 4
    model_dim = 4
    
    # 创建一个模拟序列，包含实际内容和填充（填充值为0）
    # 这里假设序列前2个是实际内容，后2个是填充
    input_seq = th.tensor([[1, 2, 0, 0]])  # 0表示填充标记
    padding_idx = 0
    
    # 2. 创建pad_mask
    # 首先创建布尔掩码，标记哪些位置是填充的
    pad_bool = th.eq(input_seq, padding_idx)  # 形状: [batch_size, seq_len]
    print(f"\n输入序列: {input_seq.numpy()}")
    print(f"填充布尔掩码: {pad_bool.numpy()}")
    
    # 扩展维度以适应注意力机制的需求
    # 最终形状: [batch_size, 1, 1, seq_len]
    pad_mask = pad_bool.unsqueeze(1).unsqueeze(2)
    print(f"扩展后的pad_mask形状: {pad_mask.shape}")
    
    # 3. 创建查询、键、值张量（模拟encoder-decoder注意力中的情况）
    # 解码器输出（查询）
    decoder_output = th.randn(batch_size, seq_len, model_dim)
    # 编码器输出（键和值）
    encoder_output = th.randn(batch_size, seq_len, model_dim)
    
    # 4. 初始化简化的注意力机制
    attention = SimplifiedMultiHeadAttention(model_dim=model_dim)
    
    print("\n===== 不使用pad_mask的情况 =====")
    _, attn_without_mask = attention.scaled_dot_product_attention(
        decoder_output, encoder_output, encoder_output)
    
    print("\n===== 使用pad_mask的情况 =====")
    _, attn_with_mask = attention.scaled_dot_product_attention(
        decoder_output, encoder_output, encoder_output, pad_mask)
    
    # 5. 可视化注意力权重
    visualize_attention(attn_without_mask, attn_with_mask)
    
    print("\n=== pad_mask在Transformer中的重要性 ===")
    print("1. 为什么需要pad_mask:")
    print("   - 防止模型关注输入序列中的填充部分")
    print("   - 避免填充位置的噪声影响模型预测")
    print("   - 确保注意力集中在实际内容上")
    
    print("\n2. pad_mask在decoder中的作用:")
    print("   - 在encoder-decoder注意力层中，防止解码器关注编码器输入中的填充部分")
    print("   - 确保生成的内容只依赖于输入序列中的有效信息")
    print("   - 提高模型性能和训练稳定性")

# 可视化有无pad_mask的注意力权重差异
def visualize_attention(attn_without_mask, attn_with_mask):
    plt.figure(figsize=(12, 5))
    
    # 绘制无掩码的注意力权重
    plt.subplot(1, 2, 1)
    plt.imshow(attn_without_mask.squeeze().detach().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('注意力权重 (无pad_mask)')
    plt.xlabel('编码器位置')
    plt.ylabel('解码器位置')
    
    # 绘制有掩码的注意力权重
    plt.subplot(1, 2, 2)
    plt.imshow(attn_with_mask.squeeze().detach().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('注意力权重 (有pad_mask)')
    plt.xlabel('编码器位置')
    plt.ylabel('解码器位置')
    
    plt.tight_layout()
    plt.savefig('pad_mask_attention_visualization.png')
    print("\n注意力权重可视化已保存为 pad_mask_attention_visualization.png")
    print("注意观察pad_mask如何将填充位置的注意力权重设为接近0")

if __name__ == "__main__":
    demonstrate_pad_mask()