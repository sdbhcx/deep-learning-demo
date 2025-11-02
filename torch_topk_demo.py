#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch中的Top-k操作详解

本脚本演示了PyTorch中topk函数的用法、参数解释、
各种应用场景以及实际使用示例。
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def main():
    """
    主函数，演示torch.topk的各种用法和应用场景
    """
    print("="*80)
    print("PyTorch中的Top-k操作详解")
    print("="*80)
    
    # 1. Top-k基本概念介绍
    print("\n1. Top-k基本概念:")
    print("   Top-k是一种选择操作，用于从一组数据中选取最大的k个元素及其索引。")
    print("   在深度学习中，Top-k常用于多分类任务的预测分析、概率分布分析等场景。")
    
    # 2. torch.topk函数参数解析
    print("\n2. torch.topk函数参数解析:")
    print("   torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None)")
    print("   参数说明:")
    print("   - input: 输入张量")
    print("   - k: 要选取的元素数量")
    print("   - dim: 在哪个维度上执行topk操作，默认为最后一个维度")
    print("   - largest: True表示选取最大的k个元素，False表示选取最小的k个元素")
    print("   - sorted: 是否按大小排序返回结果，默认为True")
    print("   返回值: (values, indices)，分别为选取的元素值和它们在原张量中的索引")
    
    # 3. 一维张量的Top-k操作示例
    print("\n3. 一维张量的Top-k操作示例:")
    # 创建一个一维张量
    scores = torch.tensor([0.1, 0.5, 0.3, 0.8, 0.2, 0.7])
    print(f"   原始张量: {scores}")
    
    # 选取最大的3个元素
    values, indices = torch.topk(scores, k=3)
    print(f"   Top-3值: {values}")
    print(f"   对应索引: {indices}")
    
    # 选取最小的2个元素
    min_values, min_indices = torch.topk(scores, k=2, largest=False)
    print(f"   Bottom-2值: {min_values}")
    print(f"   对应索引: {min_indices}")
    
    # 4. 多维张量的Top-k操作示例
    print("\n4. 多维张量的Top-k操作示例:")
    # 创建一个二维张量 (batch_size=2, num_classes=5)
    logits = torch.tensor([
        [0.1, 0.8, 0.2, 0.5, 0.3],
        [0.9, 0.4, 0.6, 0.2, 0.7]
    ])
    print(f"   原始二维张量 (批次大小=2，类别数=5):\n{logits}")
    
    # 在最后一个维度上选取Top-3
    topk_values, topk_indices = torch.topk(logits, k=3, dim=1)
    print(f"   每个样本的Top-3值 (dim=1):\n{topk_values}")
    print(f"   对应的类别索引:\n{topk_indices}")
    
    # 在第一个维度上选取Top-1
    topk_batch, topk_batch_indices = torch.topk(logits, k=1, dim=0)
    print(f"   每个位置的最大批次值 (dim=0):\n{topk_batch.squeeze()}")
    print(f"   对应的批次索引:\n{topk_batch_indices.squeeze()}")
    
    # 5. Top-k在多分类任务中的应用
    print("\n5. Top-k在多分类任务中的应用:")
    # 模拟模型输出的logits
    model_outputs = torch.tensor([
        [0.2, 0.1, 0.6, 0.1],  # 样本1的类别概率
        [0.5, 0.3, 0.1, 0.1]   # 样本2的类别概率
    ])
    
    # 计算Top-1准确率 (假设真实标签为[2, 0])
    true_labels = torch.tensor([2, 0])
    
    # 获取每个样本的最高概率类别
    _, predicted_top1 = torch.max(model_outputs, dim=1)
    top1_accuracy = (predicted_top1 == true_labels).float().mean()
    print(f"   模型输出概率:\n{model_outputs}")
    print(f"   真实标签: {true_labels}")
    print(f"   Top-1预测: {predicted_top1}")
    print(f"   Top-1准确率: {top1_accuracy:.2f}")
    
    # 计算Top-2准确率
    _, predicted_top2 = torch.topk(model_outputs, k=2, dim=1)
    # 检查真实标签是否在Top-2预测中
    top2_correct = torch.sum(
        (predicted_top2 == true_labels.unsqueeze(1)).any(dim=1)
    )
    top2_accuracy = top2_correct / len(true_labels)
    print(f"   Top-2预测: {predicted_top2}")
    print(f"   Top-2准确率: {top2_accuracy:.2f}")
    
    # 6. Top-k在特征选择中的应用
    print("\n6. Top-k在特征选择中的应用:")
    # 创建特征重要性张量 (例如从模型中提取的特征重要性)
    feature_importance = torch.tensor([0.12, 0.85, 0.34, 0.67, 0.23, 0.45, 0.76, 0.54])
    n_features = len(feature_importance)
    
    # 选取重要性最高的k个特征
    k = 3
    importance_values, feature_indices = torch.topk(feature_importance, k=k)
    print(f"   特征重要性: {feature_importance}")
    print(f"   Top-{k}重要特征的索引: {feature_indices}")
    print(f"   对应的重要性值: {importance_values}")
    
    # 7. Top-k在概率分布分析中的应用
    print("\n7. Top-k在概率分布分析中的应用:")
    # 模拟一个概率分布
    probabilities = torch.softmax(torch.randn(10), dim=0)
    print(f"   概率分布: {probabilities}")
    
    # 分析Top-3概率的累积和
    topk_probs, topk_indices = torch.topk(probabilities, k=3)
    cumulative_prob = torch.sum(topk_probs)
    print(f"   Top-3概率值: {topk_probs}")
    print(f"   Top-3概率的累积和: {cumulative_prob:.4f}")
    print(f"   占总概率的比例: {cumulative_prob * 100:.2f}%")
    
    # 8. Top-k的可视化示例
    print("\n8. Top-k的可视化示例:")
    visualize_topk()
    
    # 9. 常见错误和注意事项
    print("\n9. 常见错误和注意事项:")
    print("   9.1 注意事项:")
    print("      - k值不能大于操作维度的大小，否则会报错")
    print("      - dim参数必须有效，否则会在默认维度（通常是最后一个）上操作")
    print("      - 对于大张量，要注意内存消耗")
    
    print("   9.2 常见错误示例:")
    try:
        # 错误示例：k值过大
        torch.topk(torch.tensor([1, 2, 3]), k=5)
    except Exception as e:
        print(f"      错误示例1（k值过大）: {type(e).__name__}: {e}")
    
    try:
        # 错误示例：无效的dim参数
        torch.topk(torch.tensor([[1, 2], [3, 4]]), k=1, dim=2)
    except Exception as e:
        print(f"      错误示例2（无效dim）: {type(e).__name__}: {e}")
    
    # 10. 总结
    print("\n10. 总结:")
    print("   - Top-k操作是选取张量中最大（或最小）的k个元素及其索引的重要工具")
    print("   - 在深度学习中广泛应用于多分类准确率计算、特征选择、概率分布分析等场景")
    print("   - 使用时需注意k值的合理性和操作维度的选择")
    print("   - 可以通过largest参数控制选取最大值还是最小值")
    print("   - 可以通过sorted参数控制返回结果是否排序")

def visualize_topk():
    """
    可视化Top-k操作的结果
    """
    # 创建示例数据
    categories = list('ABCDEFGHIJKLMNOPQRST')
    scores = torch.randn(len(categories))
    
    # 计算Top-5
    k = 5
    topk_values, topk_indices = torch.topk(scores, k=k)
    
    # 创建可视化数据
    x = np.arange(len(categories))
    colors = ['gray' if i not in topk_indices else 'red' for i in range(len(categories))]
    
    # 绘制条形图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, scores.numpy(), color=colors)
    
    # 添加标签和标题
    plt.xlabel('类别')
    plt.ylabel('分数')
    plt.title(f'Top-{k} 结果可视化（红色表示Top-{k}元素）')
    plt.xticks(x, categories)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存图表（不显示，因为在命令行环境中）
    plt.tight_layout()
    plt.savefig('topk_visualization.png')
    print("   Top-k可视化图表已保存为 'topk_visualization.png'")

if __name__ == "__main__":
    main()