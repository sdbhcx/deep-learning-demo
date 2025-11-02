# -*- coding: utf-8 -*-
import torch
import sys
import io

# 确保中文正常显示
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
这个文件用于详细解释torch.norm函数的含义、参数和使用场景
"""

# 1. torch.norm函数的基本介绍
print("=== torch.norm函数基本介绍 ===")
print("torch.norm函数用于计算张量的范数（norm）")
print("官方文档定义: torch.norm(input, p='fro', dim=None, keepdim=False, out=None, dtype=None) → Tensor")
print()

# 2. 参数说明
print("=== 参数说明 ===")
print("- input: 输入张量")
print("- p: 范数的类型，可以是整数、浮点数、inf、-inf或'fro'（弗罗贝尼乌斯范数，仅适用于矩阵）")
print("- dim: 计算范数的维度，可以是整数或元组")
print("- keepdim: 是否保持输出张量的维度，默认为False")
print("- out: 可选的输出张量")
print("- dtype: 可选的输出数据类型")
print()

# 3. 基本示例 - 向量范数
print("=== 基本示例 - 向量范数 ===")

# 创建一个一维张量（向量）
vector = torch.tensor([3.0, 4.0])
print(f"向量: {vector}")
print()

# 计算L1范数（曼哈顿距离）
l1_norm = torch.norm(vector, p=1)
print(f"L1范数 (p=1): {l1_norm}")
print(f"计算方式: |3| + |4| = 7.0")
print()

# 计算L2范数（欧几里得距离）
l2_norm = torch.norm(vector, p=2)
print(f"L2范数 (p=2): {l2_norm}")
print(f"计算方式: √(3² + 4²) = 5.0")
print()

# 计算无穷范数（切比雪夫距离）
inf_norm = torch.norm(vector, p=float('inf'))
print(f"无穷范数 (p=inf): {inf_norm}")
print(f"计算方式: max(|3|, |4|) = 4.0")
print()

# 计算p=3的范数
p3_norm = torch.norm(vector, p=3)
print(f"p=3的范数: {p3_norm}")
print(f"计算方式: (|3|³ + |4|³)^(1/3) = (27 + 64)^(1/3) = 91^(1/3)")
print()

# 4. 矩阵范数示例
print("=== 矩阵范数示例 ===")

# 创建一个二维张量（矩阵）
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"矩阵:\n{matrix}")
print()

# 计算弗罗贝尼乌斯范数（类似向量的L2范数）
fro_norm = torch.norm(matrix, p='fro')
print(f"弗罗贝尼乌斯范数 (p='fro'): {fro_norm}")
print(f"计算方式: √(1² + 2² + 3² + 4²) = √30")
print()

# 计算矩阵的L1范数（列和的最大值）
matrix_l1_norm = torch.norm(matrix, p=1)
print(f"矩阵L1范数 (p=1): {matrix_l1_norm}")
print(f"计算方式: max(1+3, 2+4) = max(4, 6) = 6")
print()

# 计算矩阵的无穷范数（行和的最大值）
matrix_inf_norm = torch.norm(matrix, p=float('inf'))
print(f"矩阵无穷范数 (p=inf): {matrix_inf_norm}")
print(f"计算方式: max(1+2, 3+4) = max(3, 7) = 7")
print()

# 计算矩阵的L2范数（谱范数，即最大奇异值）
matrix_l2_norm = torch.norm(matrix, p=2)
print(f"矩阵L2范数 (p=2, 谱范数): {matrix_l2_norm}")
print("计算方式: 矩阵的最大奇异值")
print()

# 5. 在特定维度上计算范数
print("=== 在特定维度上计算范数 ===")

# 创建一个三维张量
tensor_3d = torch.tensor([
    [[1.0, 2.0], [3.0, 4.0]],
    [[5.0, 6.0], [7.0, 8.0]]
])
print(f"三维张量形状: {tensor_3d.shape}")
print(f"三维张量:\n{tensor_3d}")
print()

# 在dim=2上计算L2范数
norm_dim2 = torch.norm(tensor_3d, p=2, dim=2)
print(f"在dim=2上计算L2范数的结果:\n{norm_dim2}")
print(f"结果形状: {norm_dim2.shape}")
print()

# 在dim=(1,2)上计算L2范数
norm_dim12 = torch.norm(tensor_3d, p=2, dim=(1, 2))
print(f"在dim=(1,2)上计算L2范数的结果:\n{norm_dim12}")
print(f"结果形状: {norm_dim12.shape}")
print()

# 使用keepdim=True保持维度
norm_keepdim = torch.norm(tensor_3d, p=2, dim=2, keepdim=True)
print(f"使用keepdim=True在dim=2上计算L2范数的结果:\n{norm_keepdim}")
print(f"结果形状: {norm_keepdim.shape}")
print()

# 6. 数据类型转换
print("=== 数据类型转换 ===")

# 创建一个整数张量
int_tensor = torch.tensor([1, 2, 3, 4])
print(f"整数张量: {int_tensor}, 数据类型: {int_tensor.dtype}")

# 将整数张量转换为浮点型张量
float_tensor = int_tensor.float()
print(f"浮点型张量: {float_tensor}, 数据类型: {float_tensor.dtype}")

# 指定输出数据类型为float64
norm_float64 = torch.norm(float_tensor, dtype=torch.float64)
print(f"指定dtype=torch.float64的范数结果: {norm_float64}")
print(f"结果数据类型: {norm_float64.dtype}")
print()

# 7. 实际应用场景
print("=== 实际应用场景 ===")
print("torch.norm在以下场景特别有用:")
print("1. 特征标准化: 将特征向量缩放到单位长度")
print("2. 损失函数计算: 如L1、L2正则化项")
print("3. 梯度裁剪: 防止梯度爆炸")
print("4. 相似度计算: 如余弦相似度需要计算向量的L2范数")
print("5. 矩阵分析: 评估矩阵的条件数、稳定性等")
print()

# 8. 特征标准化示例
print("=== 特征标准化示例 ===")

# 创建特征向量
features = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(f"原始特征:\n{features}")

# 计算每个样本的L2范数
norms = torch.norm(features, p=2, dim=1, keepdim=True)
print(f"每个样本的L2范数:\n{norms}")

# 标准化特征
normalized_features = features / norms
print(f"标准化后的特征:\n{normalized_features}")

# 验证标准化后的特征范数为1
normalized_norms = torch.norm(normalized_features, p=2, dim=1)
print(f"标准化后的特征范数:\n{normalized_norms}")
print()

# 9. 梯度裁剪示例
print("=== 梯度裁剪示例 ===")

# 创建一个需要梯度的张量
params = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 模拟前向传播和反向传播
output = params.norm(p=2)
output.backward()

print(f"原始梯度: {params.grad}")

# 计算梯度范数
grad_norm = torch.norm(params.grad, p=2)
print(f"梯度范数: {grad_norm}")

# 梯度裁剪阈值
clip_value = 2.0

# 如果梯度范数超过阈值，则进行裁剪
if grad_norm > clip_value:
    clipped_grad = params.grad * (clip_value / grad_norm)
    print(f"裁剪后的梯度: {clipped_grad}")
print()

# 10. 常见错误和注意事项
print("=== 常见错误和注意事项 ===")
print("1. p='fro'只能用于二维矩阵，不适用于一维向量或更高维张量")
print("2. 当计算无穷范数时，使用float('inf')而不是字符串'inf'")
print("3. 确保在计算范数之前张量的数据类型适合（通常为浮点型）")
print("4. 在高维张量上计算范数时，注意指定正确的维度")
print("5. 范数计算可能会受到数值精度的影响，特别是对于非常大或非常小的值")
print()

# 11. 总结
print("=== 总结 ===")
print("torch.norm函数的主要功能:")
print("- 计算张量的各种类型的范数，包括向量范数和矩阵范数")
print("- 支持在特定维度或维度组合上计算范数")
print("- 可用于特征标准化、损失函数计算、梯度裁剪等多种深度学习场景")
print("- 提供了灵活的参数选项，如keepdim和dtype，以适应不同的计算需求")