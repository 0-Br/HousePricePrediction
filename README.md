# HousePricePrediction

基于深度学习的北京二手房价格预测 | Beijing Second-Hand House Price Prediction with Deep Learning

清华大学吴璟老师「工程经济学」课程研究项目

刘滨瑞 / Binrui Liu, Tsinghua University

## 项目简介 / Overview

本项目使用深度学习方法对北京市二手房交易价格进行时间序列预测。数据来源于链家（Lianjia）平台约 31.9 万条真实成交记录，涵盖 2014–2018 年的日均价格序列。项目实现了基于一维卷积神经网络（1D-CNN）和长短期记忆网络（LSTM）的两种预测模型，并通过 Visdom 实时可视化训练过程。此外还收集了 PM2.5 数据与 13 项中国宏观经济指标（CPI、PMI、货币供应量、存贷款利率、社会融资规模等），为后续多因素建模提供数据支撑。

This project applies deep learning to time-series prediction of second-hand housing prices in Beijing. The dataset comprises approximately 319,000 real transaction records from the Lianjia platform, covering daily average prices from 2014 to 2018. Two prediction models are implemented — a 1D Convolutional Neural Network (1D-CNN) and a Long Short-Term Memory network (LSTM) — with Visdom providing real-time training visualization. Additionally, PM2.5 data and 13 macroeconomic indicators (CPI, PMI, money supply, deposit/loan interest rates, aggregate financing, etc.) were collected to support potential multi-factor modeling.

## 数据说明 / Data

### 房价数据 / Housing Price Data

| 文件 / File | 说明 / Description |
|---|---|
| `Data/HousePrice_Peking.csv` | 北京二手房链家成交记录（约 31.9 万条），含经纬度、成交价、面积、楼层、建筑年代等 / ~319k Lianjia transaction records with coordinates, price, area, floor, construction year, etc. |
| `Data/D_HousePrice_Peking.csv` | 预处理后的日均价格序列（2014-01-01 起） / Preprocessed daily average price series (from 2014-01-01) |
| `Data/HousePrice_Boston.csv` | 波士顿房价经典数据集（对照参考） / Boston housing dataset (baseline reference) |

### 环境与宏观数据 / Environmental & Macroeconomic Data

| 文件 / File | 说明 / Description |
|---|---|
| `Data/PM2.5_Peking.csv` | 北京 PM2.5 监测数据（2010 年起） / Beijing PM2.5 monitoring data (from 2010) |
| `Data/CN/` | 中国宏观经济指标：CPI、PMI、人口、存贷款利率、货币供应量、社会融资规模、税收、进出口等 / Chinese macroeconomic indicators: CPI, PMI, population, interest rates, money supply, aggregate financing, taxation, trade, etc. |

## 模型结构 / Model Architecture

### 1D-CNN (`CNNpredict.py`)

| 项目 / Item | 配置 / Configuration |
|---|---|
| 输入 / Input | 过去 360 天日均价格滑动窗口 / 360-day sliding window of daily average prices |
| 结构 / Architecture | 两层 1D 卷积 + MaxPool → 全连接层 / Two 1D-Conv layers + MaxPool → Fully connected |
| 损失函数 / Loss | Smooth L1 Loss (Huber Loss) |
| 优化器 / Optimizer | Adam (lr = 1e-5) |
| 训练轮次 / Epochs | 100 |
| 预测 / Prediction | 向前滚动预测 180 天 / 180-day rolling forecast |

### LSTM (`RNNpredict.py`)

| 项目 / Item | 配置 / Configuration |
|---|---|
| 输入 / Input | 过去 7 天日均价格滑动窗口 / 7-day sliding window of daily average prices |
| 结构 / Architecture | 两层 LSTM → 线性回归层 / Two LSTM layers → Linear regression |
| 损失函数 / Loss | MSE Loss |
| 优化器 / Optimizer | Adam (lr = 1e-5) |
| 训练轮次 / Epochs | 100,000 |
| 预测 / Prediction | 对整段时间序列拟合与预测 / Fitting and prediction over the full time series |

## 运行方式 / Usage

### 环境依赖 / Dependencies

- Python 3
- NumPy, Pandas, scikit-learn, Matplotlib
- PyTorch
- Visdom

### 步骤 / Steps

```bash
# 启动 Visdom 可视化服务 / Start Visdom visualization server
python -m visdom.server

# 预处理数据 / Preprocess data
python pre_processing.py

# 运行 CNN 预测 / Run CNN prediction
python CNNpredict.py

# 运行 LSTM 预测 / Run LSTM prediction
python RNNpredict.py
```

## 项目结构 / Project Structure

```
HousePricePrediction/
├── pre_processing.py    # 数据预处理 / Data preprocessing
├── CNNpredict.py        # 1D-CNN 预测模型 / 1D-CNN prediction model
├── RNNpredict.py        # LSTM 预测模型 / LSTM prediction model
├── Data/
│   ├── HousePrice_Peking.csv
│   ├── D_HousePrice_Peking.csv
│   ├── HousePrice_Boston.csv
│   ├── PM2.5_Peking.csv
│   └── CN/              # 宏观经济指标 / Macroeconomic indicators
└── README.md
```
