import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

import numpy as np
from pandas.io.parsers import read_csv
#pytorch version2

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x)

# 4가지 변수를 입력 받습니다.
avg_temp = float(input('평균 온도: '))
min_temp = float(input('최저 온도: '))
max_temp = float(input('최고 온도: '))
rain_fall = float(input('강수량: '))

# 모델 초기화
model = MultivariateLinearRegressionModel()
model.load_state_dict(torch.load('./saved.pt'))
model.eval()

data = (avg_temp, min_temp, max_temp, rain_fall)
print(data)
arr = np.array(data, dtype=np.float32)
print(arr, arr.shape)

x_data = arr[0:4]
print(x_data, x_data.shape)

x_infer = torch.FloatTensor(x_data)

y_predict = model(x_infer)
print(y_predict.shape)

print("- 예상 배추가격 : ", y_predict.squeeze().detach().numpy())