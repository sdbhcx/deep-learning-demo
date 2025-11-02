# -*- coding: utf-8 -*-
import torch
import sys
import io

# 确保中文正常显示
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
这个文件用于解释 torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1) 的含义和作用
"""

# 1. 分解解释每个函数的作用
print("=== 分解解释 ===")

# torch.linspace(-1, 1, 1000) 的作用
print("\ntorch.linspace(-1, 1, 1000) 函数:")
print("- 作用: 创建一个一维张量，包含从-1到1的等间隔的1000个数值")
print("- 第一个参数: 起始值 (-1)")
print("- 第二个参数: 结束值 (1)")
print("- 第三个参数: 元素数量 (1000)")

# 创建linspace张量并查看其形状
linspace_tensor = torch.linspace(-1, 1, 1000)
print(f"- 生成的张量形状: {linspace_tensor.shape}")
print(f"- 前5个元素: {linspace_tensor[:5]}")
print(f"- 后5个元素: {linspace_tensor[-5:]}")

# torch.unsqueeze(dim=1) 的作用
print("\ntorch.unsqueeze(dim=1) 函数:")
print("- 作用: 在指定维度上增加一个维度")
print("- dim=1: 表示在第2个维度(索引为1)的位置增加一个维度")
print("- 常用于将一维数据转换为二维数据，以便与PyTorch的神经网络层兼容")

# 应用unsqueeze并查看结果形状
unsqueeze_tensor = torch.unsqueeze(linspace_tensor, dim=1)
print(f"- 转换后的张量形状: {unsqueeze_tensor.shape}")
print(f"- 前5个元素:\n{unsqueeze_tensor[:5]}")

# 2. 整体代码的作用和应用场景
print("\n=== 整体代码的作用 ===")
print("torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1) 整体作用:")
print("1. 首先创建一个包含1000个等间隔数值的一维张量")
print("2. 然后将其转换为形状为 (1000, 1) 的二维张量")
print("3. 这种形状通常用于表示包含1000个样本，每个样本有1个特征的数据集")

# 3. 在深度学习中的应用场景
print("\n=== 在深度学习中的应用场景 ===")
print("这种操作在以下场景特别有用:")
print("1. 准备训练数据: 将一维输入数据转换为二维格式，符合PyTorch模型的输入要求")
print("2. 特征扩展: 为数据添加新的维度，便于后续的矩阵运算")
print("3. 批量处理: 配合DataLoader使用，可以方便地进行批量训练")

# 4. 等价写法示例
print("\n=== 等价写法示例 ===")
print("以下几种写法与原代码效果相同:")

equivalent_1 = linspace_tensor.view(-1, 1)
equivalent_2 = linspace_tensor.reshape(-1, 1)
equivalent_3 = linspace_tensor.unsqueeze(1)  # 可以省略dim参数名

print(f"使用view方法: {equivalent_1.shape}")
print(f"使用reshape方法: {equivalent_2.shape}")
print(f"省略dim参数名: {equivalent_3.shape}")

# 5. 实际应用示例
print("\n=== 实际应用示例 ===")

# 创建一个简单的数据集
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1 * torch.randn(x.size())  # 创建y = x² + 噪声

print(f"输入数据形状: {x.shape}")
print(f"输出数据形状: {y.shape}")
print(f"这表示有1000个样本，每个样本有1个输入特征和1个输出值")

# 模拟简单的神经网络输入
print("\n模拟神经网络的第一层处理:")
# 假设我们有一个简单的线性层，输入维度为1，输出维度为10
linear_layer = torch.nn.Linear(1, 10)
output = linear_layer(x)
print(f"线性层输出形状: {output.shape}")
print("这表明成功将输入数据传递给了神经网络层")

# 6. 总结
print("\n=== 总结 ===")
print("torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1) 的主要作用是:")
print("- 创建一个形状为 (1000, 1) 的二维张量")
print("- 这种形状符合PyTorch中大多数神经网络层的输入要求")
print("- 使得一维数据能够被正确地用于深度学习模型的训练和预测")