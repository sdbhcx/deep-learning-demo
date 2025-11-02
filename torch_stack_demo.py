# -*- coding: utf-8 -*-
import torch
import sys
import io

# 确保中文正常显示
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
这个文件用于详细解释torch.stack函数的含义、参数和使用场景
"""

# 1. torch.stack函数的基本介绍
print("=== torch.stack函数基本介绍 ===")
print("torch.stack函数用于沿着新的维度对输入张量序列进行连接")
print("官方文档定义: torch.stack(tensors, dim=0, *, out=None) → Tensor")
print()

# 2. 参数说明
print("=== 参数说明 ===")
print("- tensors: 一个序列的张量，所有张量必须具有相同的形状")
print("- dim: 指定在哪个维度上进行堆叠，默认为0")
print("- out: 可选的输出张量")
print()

# 3. 基本示例
print("=== 基本示例 ===")

# 创建两个形状相同的张量
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])

print("原始张量:")
print(f"tensor1: {tensor1}, 形状: {tensor1.shape}")
print(f"tensor2: {tensor2}, 形状: {tensor2.shape}")
print()

# 沿dim=0堆叠
result_dim0 = torch.stack([tensor1, tensor2], dim=0)
print(f"沿dim=0堆叠后的结果:")
print(result_dim0)
print(f"形状: {result_dim0.shape}")
print()

# 沿dim=1堆叠
result_dim1 = torch.stack([tensor1, tensor2], dim=1)
print(f"沿dim=1堆叠后的结果:")
print(result_dim1)
print(f"形状: {result_dim1.shape}")
print()

# 4. 二维张量的堆叠示例
print("=== 二维张量的堆叠示例 ===")

# 创建两个形状相同的二维张量
tensor3 = torch.tensor([[1, 2], [3, 4]])
tensor4 = torch.tensor([[5, 6], [7, 8]])

print("原始二维张量:")
print(f"tensor3:\n{tensor3}, 形状: {tensor3.shape}")
print(f"tensor4:\n{tensor4}, 形状: {tensor4.shape}")
print()

# 沿dim=0堆叠
result_2d_dim0 = torch.stack([tensor3, tensor4], dim=0)
print(f"沿dim=0堆叠二维张量的结果:")
print(result_2d_dim0)
print(f"形状: {result_2d_dim0.shape}")
print()

# 沿dim=1堆叠
result_2d_dim1 = torch.stack([tensor3, tensor4], dim=1)
print(f"沿dim=1堆叠二维张量的结果:")
print(result_2d_dim1)
print(f"形状: {result_2d_dim1.shape}")
print()

# 沿dim=2堆叠
result_2d_dim2 = torch.stack([tensor3, tensor4], dim=2)
print(f"沿dim=2堆叠二维张量的结果:")
print(result_2d_dim2)
print(f"形状: {result_2d_dim2.shape}")
print()

# 5. 形状变化可视化
print("=== 形状变化可视化 ===")
print("当堆叠n个形状为(a, b, c)的张量时:")
print("- 沿dim=0堆叠后形状变为: (n, a, b, c)")
print("- 沿dim=1堆叠后形状变为: (a, n, b, c)")
print("- 沿dim=2堆叠后形状变为: (a, b, n, c)")
print("- 沿dim=3堆叠后形状变为: (a, b, c, n)")
print()

# 6. 与其他连接函数的区别
print("=== 与其他连接函数的区别 ===")

# 创建示例张量
tensor5 = torch.tensor([[1, 2], [3, 4]])
tensor6 = torch.tensor([[5, 6], [7, 8]])

# 使用torch.cat (连接)
result_cat = torch.cat([tensor5, tensor6], dim=0)
print(f"torch.cat沿dim=0的结果 (形状: {result_cat.shape}):")
print(result_cat)
print()

# 使用torch.stack (堆叠)
result_stack = torch.stack([tensor5, tensor6], dim=0)
print(f"torch.stack沿dim=0的结果 (形状: {result_stack.shape}):")
print(result_stack)
print()

print("区别总结:")
print("1. torch.cat: 在现有维度上连接张量，不增加新维度")
print("2. torch.stack: 沿着新维度堆叠张量，增加一个新维度")
print("3. torch.cat要求张量除了连接维度外，其他维度必须相同")
print("4. torch.stack要求所有张量形状必须完全相同")
print()

# 7. 实际应用场景
print("=== 实际应用场景 ===")
print("torch.stack在以下场景特别有用:")
print("1. 数据预处理: 将多个样本合并成一个批次")
print("2. 特征组合: 沿着新维度组合不同的特征表示")
print("3. 时间序列数据处理: 将多个时间步的数据堆叠成时间维度")
print("4. 多模态数据融合: 沿新维度融合不同模态的数据")
print()

# 8. 批量数据处理示例
print("=== 批量数据处理示例 ===")

# 创建3个形状为(2, 3)的样本
sample1 = torch.randn(2, 3)
sample2 = torch.randn(2, 3)
sample3 = torch.randn(2, 3)

print(f"单个样本形状: {sample1.shape}")

# 使用stack创建批次
batch = torch.stack([sample1, sample2, sample3], dim=0)
print(f"堆叠后的批次形状: {batch.shape}")
print("这表示一个批次包含3个样本，每个样本形状为(2, 3)")
print()

# 9. 常见错误和注意事项
print("=== 常见错误和注意事项 ===")
print("1. 所有输入张量必须具有完全相同的形状，否则会抛出错误")
print("2. dim参数必须在有效范围内，即[-len(result_shape), len(result_shape)-1]")
print("3. 堆叠操作会增加内存使用，因为它创建了一个新的张量")
print("4. 对于大型张量，考虑是否真的需要堆叠，或者可以使用更内存高效的方法")
print()

# 10. 总结
print("=== 总结 ===")
print("torch.stack函数的主要功能:")
print("- 沿着新的维度对形状相同的张量序列进行堆叠")
print("- 结果张量的维度比输入张量多一维")
print("- 是构建批次数据、组合多源信息的重要工具")
print("- 与torch.cat的主要区别在于是否增加新维度")