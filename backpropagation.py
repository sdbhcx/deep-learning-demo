# -*- coding: utf-8 -*-
import numpy as np
import sys
import io

# 确保中文正常显示
if sys.platform.startswith('win'):
    # 在Windows系统上设置标准输出为UTF-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class NeuralNetwork:
    """
    一个简单的神经网络类，用于手写实现反向传播算法
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        初始化神经网络参数
        
        参数:
            input_size: 输入层神经元数量
            hidden_size: 隐藏层神经元数量
            output_size: 输出层神经元数量
            learning_rate: 学习率，默认为0.01
        """
        # 随机初始化权重和偏置
        self.weights1 = np.random.randn(input_size, hidden_size)  # 输入层到隐藏层的权重
        self.bias1 = np.random.randn(1, hidden_size)  # 隐藏层的偏置
        self.weights2 = np.random.randn(hidden_size, output_size)  # 隐藏层到输出层的权重
        self.bias2 = np.random.randn(1, output_size)  # 输出层的偏置
        self.learning_rate = learning_rate
        
    def sigmoid(self, x):
        """
        Sigmoid激活函数
        
        参数:
            x: 输入值
        
        返回:
            应用sigmoid后的输出值
        """
        return 1 / (1 + np.exp(-x))
        
    def sigmoid_derivative(self, x):
        """
        Sigmoid激活函数的导数
        
        参数:
            x: 输入值（注意这里x应该是已经经过sigmoid激活后的值）
        
        返回:
            sigmoid导数计算结果
        """
        return x * (1 - x)
        
    def forward_propagation(self, X):
        """
        前向传播过程
        
        参数:
            X: 输入数据，形状为[样本数, 输入层神经元数量]
        
        返回:
            y_pred: 预测输出
            cache: 缓存中间结果，用于反向传播
        """
        # 隐藏层计算
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.sigmoid(self.z1)
        
        # 输出层计算
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        y_pred = self.sigmoid(self.z2)
        
        # 缓存中间结果
        cache = (X, self.z1, self.a1, self.z2, y_pred)
        
        return y_pred, cache
        
    def compute_loss(self, y_pred, y_true):
        """
        计算均方误差损失
        
        参数:
            y_pred: 预测输出
            y_true: 真实标签
        
        返回:
            损失值
        """
        return np.mean(np.square(y_pred - y_true))
        
    def backpropagation(self, cache, y_true):
        """
        反向传播过程，计算梯度并更新权重
        
        参数:
            cache: 前向传播过程中缓存的中间结果
            y_true: 真实标签
        """
        X, z1, a1, z2, y_pred = cache
        m = X.shape[0]  # 样本数量
        
        # 输出层误差
        dz2 = (y_pred - y_true) * self.sigmoid_derivative(y_pred)
        
        # 隐藏层到输出层权重和偏置的梯度
        dweights2 = np.dot(a1.T, dz2) / m
        dbias2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # 隐藏层误差
        dz1 = np.dot(dz2, self.weights2.T) * self.sigmoid_derivative(a1)
        
        # 输入层到隐藏层权重和偏置的梯度
        dweights1 = np.dot(X.T, dz1) / m
        dbias1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # 更新权重和偏置
        self.weights1 -= self.learning_rate * dweights1
        self.bias1 -= self.learning_rate * dbias1
        self.weights2 -= self.learning_rate * dweights2
        self.bias2 -= self.learning_rate * dbias2
        
    def train(self, X, y, epochs=10000, verbose=True):
        """
        训练神经网络
        
        参数:
            X: 输入数据
            y: 真实标签
            epochs: 训练轮数
            verbose: 是否打印训练过程
        """
        for epoch in range(epochs):
            # 前向传播
            y_pred, cache = self.forward_propagation(X)
            
            # 计算损失
            loss = self.compute_loss(y_pred, y)
            
            # 反向传播和参数更新
            self.backpropagation(cache, y)
            
            # 每1000轮打印一次损失
            if verbose and epoch % 1000 == 0:
                print(f"轮次 {epoch}, 损失值: {loss:.6f}")

# 测试代码
if __name__ == "__main__":
    # 创建一个简单的XOR问题数据集
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    print("XOR问题的数据集:")
    print("输入:", X)
    print("输出:", y)
    
    # 创建神经网络实例
    input_size = 2
    hidden_size = 4
    output_size = 1
    model = NeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.1)
    
    print("\n开始训练神经网络...")
    model.train(X, y, epochs=10000)
    
    # 测试模型
    print("\n测试模型预测结果:")
    y_pred, _ = model.forward_propagation(X)
    print("原始预测:", y_pred)
    print("四舍五入后:", np.round(y_pred))
    
    # 可视化权重更新过程（简单展示最终权重）
    print("\n最终权重和偏置:")
    print("输入层到隐藏层权重:", model.weights1)
    print("隐藏层偏置:", model.bias1)
    print("隐藏层到输出层权重:", model.weights2)
    print("输出层偏置:", model.bias2)

    # 生成训练数据
# torch.unsqueeze() 的作用是将一维变二维，torch只能处理二维的数据
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)  
# 0.1 * torch.normal(x.size())增加噪点
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

torch_dataset = Data.TensorDataset(x,y)
#得到一个代批量的生成器
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)