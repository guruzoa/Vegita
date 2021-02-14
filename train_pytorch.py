import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

import numpy as np
from pandas.io.parsers import read_csv

data = read_csv('Python Software\price_data.csv', sep=',')

xy = np.array(data, dtype=np.float32)
# print(xy.shape)

x_data = xy[:, 1:-1] 
print(x_data.shape) # N, X (2922, 4)
y_data = xy[:, [-1]]
print(y_data.shape) # N, Y (2922, 1)


x_tensor = torch.FloatTensor(x_data)
y_tensor = torch.FloatTensor(y_data)

W = torch.zeros([4, 1], requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=1e-3)

for step in range(100001):
    hypothesis = x_tensor.matmul(W) + b

    cost = torch.mean((hypothesis - y_tensor) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if step % 500 == 0:
        print("#", step, " 손실 비용: ", cost.item())
        print("- 배추가격 : ", hypothesis.squeeze().detach())

# torch.save('')
# print('saved model')