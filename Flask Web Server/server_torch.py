# -*- coding: utf-8 -*-
from flask import Flask, render_template, request

import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

app = Flask(__name__)

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x)

# 모델 초기화
model = MultivariateLinearRegressionModel()
model.load_state_dict(torch.load('./model/saved.pt'))
model.eval()

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        # 파라미터를 전달 받습니다.
        avg_temp = float(request.form['avg_temp'])
        min_temp = float(request.form['min_temp'])
        max_temp = float(request.form['max_temp'])
        rain_fall = float(request.form['rain_fall'])

        # 배추 가격 변수를 선언합니다.
        price = 0

        data = (avg_temp, min_temp, max_temp, rain_fall)
        arr = np.array(data, dtype=np.float32)
        x_data = arr[0:4]
        x_infer = torch.FloatTensor(x_data)

        y_predict = model(x_infer)
        predict = y_predict.squeeze().detach().numpy()
        print("- 예상 배추가격 : ", predict)

        # 결과 배추 가격을 저장합니다.
        price = predict

        return render_template('index.html', price=price)

if __name__ == '__main__':
   app.run(debug = True)