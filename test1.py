# -*- coding: utf-8 -*-
import numpy as np
import sys
import io
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# 确保中文正常显示
if sys.platform.startswith('win'):
    # 在Windows系统上设置标准输出为UTF-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# torch.manual_seed(10)

# loss = nn.CrossEntropyLoss(reduction='mean')
# input = torch.randn(1,2, requires_grad=True)
# print(input)
# target = torch.randn(1,2)
# print(target)
# output = loss(input, target)
# print(output)
# output.backward()
# print(input.grad)

# a = torch.tensor([[3,3]], dtype=torch.float32)
# b = torch.ones(2,2)

# print(b)
# print(torch.matmul(a,b))
print(th.from_numpy(np.expand_dims(np.arange(4), 1)).type(th.float32))