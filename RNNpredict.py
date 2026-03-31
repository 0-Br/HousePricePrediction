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
size_test = 360#测试长度
size_window = 7#训练长度7，即一周
set_train = normset[:-size_test]#训练集
set_test = normset[-size_test:]#测试集

#训练集初始化
def input_data(qe, size_window):
    windows = []
    labels = []
    length = len(qe)
    for i in range(length - size_window):
        window = qe[i : i + size_window]
        label = qe[i + size_window]
        windows.append(window)
        labels.append(label)
    return (np.array(windows), np.array(labels))
data_train = input_data(set_train, size_window)

#循环神经网络搭建
class RNNnetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size = 1, num_layers = 2):
        super(RNNnetwork, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)#RNN层
        self.reg = nn.Linear(hidden_size, output_size)#线性回归层

    def forward(self, x):
        x, _ = self.rnn(x)#(seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x

torch.manual_seed(8080)#训练种子
model = RNNnetwork(size_window, 4)
model.to(device)
func_loss = nn.MSELoss()#采用均方损失作为loss函数
optimizer = optim.Adam(model.parameters(), lr = 0.00001)#采用Adam优化，学习速率设为0.00001
epochs = 100000#训练周期为100000

#模型训练
model.train()
time_start = time.time()
print('训练开始！')
for epoch in range(epochs):
    time_0 = time.time()
    x = torch.FloatTensor(data_train[0].reshape(-1, 1, size_window))
    x = x.to(device)
    label = torch.FloatTensor(data_train[1].reshape(-1, 1, 1))
    label = label.to(device)
    optimizer.zero_grad()#每次更新参数前都梯度归零和初始化
    y = model(x)
    loss = func_loss(y, label)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0 or epoch == 0:
        print('第%d次训练：loss = %f，用时%.6fs' % (epoch + 1, loss.item(), (time.time() - time_0)))
        viz.line([loss.item()], [epoch],win = 'train_loss', update = 'append')
print('训练结束！总用时%.6fs' % (time.time() - time_start))

#预测
model.eval()#设置成eval模式
x_p = torch.FloatTensor(input_data(normset, size_window)[0].reshape(-1, 1, size_window))
x_p = x_p.to(device)
preds = model(x_p)
preds = preds.cpu()
preds = preds.view(-1, 1).detach().numpy()
true_predictions = scaler.inverse_transform(np.array(preds).reshape(-1, 1))#逆归一化，还原真实值

#对比真实值和预测值
plt.figure(figsize = (15, 3))
plt.grid(True)
plt.plot(dataset['price'], label = 'Reality')
x = np.arange('2014-01-07', '2018-01-28', dtype = 'datetime64[D]').astype('datetime64[D]')
plt.plot(x[30:], true_predictions[30:], label = 'Prediction')
plt.legend(loc = 'best')
plt.show()
