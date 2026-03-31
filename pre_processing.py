# -*- coding: utf-8 -*-

import pandas as pd
import datetime
import os
os.chdir(os.path.dirname(__file__))

dataset = pd.read_csv('Data/HousePrice_Peking.csv', index_col = 0, parse_dates = True, usecols = ['tradeTime', 'price'], encoding = 'gbk')
dataset = dataset.resample('d').mean()#按日计算平均值
dataset['price'] = round(dataset['price'].interpolate(method = 'linear', axis = 0),1)#线性内插
dataset.to_csv('Data/D_HousePrice_Peking.csv', encoding = 'gbk')

dataset = pd.read_csv('Data/D_HousePrice_Peking.csv', parse_dates = True, usecols = ['tradeTime', 'price'], encoding = 'gbk')
dataset = dataset[pd.to_datetime(dataset.tradeTime) >= datetime.datetime(2014,1,1)]
dataset.to_csv('Data/D_HousePrice_Peking.csv', encoding = 'gbk')
