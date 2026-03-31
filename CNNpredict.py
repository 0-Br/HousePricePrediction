# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from visdom import Visdom
viz = Visdom()
viz.line([0.], [0.], win = 'train_loss', opts = dict(title = 'train loss'))
#python -m visdom.server

import time
import os
os.chdir(os.path.dirname(__file__))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('使用GPU：' + str(device))

#数据导入
dataset = pd.read_csv('Data/D_HousePrice_Peking.csv', index_col = 0, parse_dates = True, usecols = ['tradeTime', 'price'], encoding = 'gbk')
dataset.sort_values('tradeTime', inplace = True)
'''
plt.figure(figsize=(12,4))
plt.grid(True)
plt.plot(dataset.sort_values('tradeTime')['price'])
plt.show()
'''

#划分训练集和测试集，将最后180个值作为测试集
scaler = MinMaxScaler(feature_range = (-1, 1))
normset = scaler.fit_transform(dataset['price'].values.astype(float).reshape(-1, 1))
size_test = 180#测试长度
size_pred = 180#预测长度
size_window = 360#训练长度360，即一年
set_train = normset[:-size_test]#训练集
set_test = normset[-size_test:]#测试集

#训练集初始化
set_train = torch.FloatTensor(set_train).view(-1)#建立一维张量
def input_data(qe, size_window):
    re0 = []
    length = len(qe)
    for i in range(length - size_window):
        window = qe[i : i + size_window]
        label = qe[i + size_window]
        re0.append((window, label))
    return re0
data_train = input_data(set_train, size_window)

#卷积神经网络搭建
class CNNnetwork(nn.Module):
    def __init__(self):
        super(CNNnetwork, self).__init__()
        self.conv1 = nn.Sequential(
                                nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 2),
                                nn.ReLU(), nn.MaxPool1d(kernel_size = 2, stride = 1))#卷积层1
        self.conv2 = nn.Sequential(
                                nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 2),
                                nn.ReLU(), nn.MaxPool1d(kernel_size = 2, stride = 1))#卷积层2
        self.relu = nn.ReLU(inplace = True)#ReLu激活函数层
        self.Linear1= nn.Linear(128 * 356, 180)#线性层1
        self.Linear2= nn.Linear(180, 1)#线性层2

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1)
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.Linear2(x)
        x = x.view(-1)
        return x

torch.manual_seed(8080)#训练种子
model = CNNnetwork()
model.to(device)
func_loss = nn.SmoothL1Loss()#采用Huber损失作为loss函数
optimizer = optim.Adam(model.parameters(), lr = 0.00001)#采用Adam优化，学习速率设为0.00001
epochs = 100#训练周期为100

#模型训练
model.train()
time_start = time.time()
print('训练开始！')
for epoch in range(epochs):
    time_0 = time.time()
    for window, label in data_train:
        label = label.reshape(-1)
        window = window.to(device)
        label = label.to(device)
        optimizer.zero_grad()#每次更新参数前都梯度归零和初始化
        y = model(window.reshape(1, 1, -1))
        loss = func_loss(y, label)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 2 == 0:
        print('第%d次训练：loss = %f，用时%.6fs' % (epoch + 1, loss.item(), (time.time() - time_0)))  
    viz.line([loss.item()], [epoch],win = 'train_loss', update = 'append')
print('训练结束！总用时%.6fs' % (time.time() - time_start))

#预测
preds = set_train[-size_window:].tolist()
model.eval()# 设置成eval模式
# 循环的每一步表示向时间序列向后滑动一格
for i in range(size_test + size_pred):
    seq = torch.FloatTensor(preds[-size_window:])
    seq = seq.to(device)
    with torch.no_grad():
        preds.append(model(seq.reshape(1,1,-1)).item())
true_predictions = scaler.inverse_transform(np.array(preds[size_window:]).reshape(-1, 1))#逆归一化，还原真实值

#对比真实值和预测值
plt.figure(figsize = (15, 3))
plt.grid(True)
plt.plot(dataset['price'], label = 'Reality')
x = np.arange('2017-08-02', '2018-07-28', dtype = 'datetime64[D]').astype('datetime64[D]')
plt.plot(x, true_predictions, label = 'Prediction')
plt.legend(loc = 'best')
plt.show()
