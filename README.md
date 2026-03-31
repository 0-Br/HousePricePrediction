# HousePricePrediction

基于深度学习的北京房价预测 | Beijing House Price Prediction with CNN & LSTM

清华大学《工程经济学》课程研究项目（吴璟老师）

## 项目简介

本项目使用深度学习方法对北京市二手房交易价格进行时间序列预测。数据来源于链家（Lianjia）平台的真实成交记录，涵盖 2014 年至 2018 年的日均价格序列。项目分别实现了基于一维卷积神经网络（1D-CNN）和长短期记忆网络（LSTM）的两种预测模型，并通过 Visdom 实时可视化训练过程。

此外，项目还收集了多项中国宏观经济指标（CPI、PMI、货币供应量、存贷款利率、社会融资规模等），为后续多因素建模提供数据支撑。

## 数据说明

### 房价数据

| 文件 | 说明 |
|------|------|
| `Data/HousePrice_Peking.csv` | 北京二手房链家成交记录（约 31.9 万条），包含经纬度、成交价、面积、楼层、建筑年代等字段 |
| `Data/D_HousePrice_Peking.csv` | 经预处理后的日均价格序列（2014-01-01 起） |
| `Data/HousePrice_Boston.csv` | 波士顿房价经典数据集（对照参考） |

### 环境与宏观数据

| 文件 | 说明 |
|------|------|
| `Data/PM2.5_Peking.csv` | 北京 PM2.5 监测数据（2010 年起） |
| `Data/CN/` | 中国宏观经济指标合集：CPI、PMI、人口、存贷款利率、货币供应量、社会融资规模、税收、进出口贸易等 |

## 模型结构

### CNN 模型 (`CNNpredict.py`)

- 输入：过去 360 天（约一年）的日均价格滑动窗口
- 结构：两层 1D 卷积 + MaxPool → 全连接层
- 损失函数：Smooth L1 Loss（Huber Loss）
- 优化器：Adam（lr = 1e-5）
- 训练轮次：100 epochs
- 预测：向前滚动预测 180 天

### LSTM 模型 (`RNNpredict.py`)

- 输入：过去 7 天的日均价格滑动窗口
- 结构：两层 LSTM → 线性回归层
- 损失函数：MSE Loss
- 优化器：Adam（lr = 1e-5）
- 训练轮次：100,000 epochs
- 预测：对整段时间序列进行拟合与预测

## 运行方式

### 环境依赖

- Python 3
- NumPy、Pandas、scikit-learn、Matplotlib
- PyTorch
- Visdom

### 步骤

```bash
# 1. 启动 Visdom 可视化服务
python -m visdom.server

# 2. 预处理数据
python pre_processing.py

# 3. 运行 CNN 预测
python CNNpredict.py

# 4. 或运行 LSTM 预测
python RNNpredict.py
```

## 项目结构

```
HousePricePrediction/
├── pre_processing.py    # 数据预处理：日均重采样、线性插值、时间筛选
├── CNNpredict.py        # 1D-CNN 房价预测模型
├── RNNpredict.py        # LSTM 房价预测模型
├── Data/
│   ├── HousePrice_Peking.csv
│   ├── D_HousePrice_Peking.csv
│   ├── HousePrice_Boston.csv
│   ├── PM2.5_Peking.csv
│   └── CN/              # 中国宏观经济指标
│       ├── CPI.xls
│       ├── PMI.xls
│       ├── 货币供应量.xls
│       ├── 存款利率.xls
│       └── ...
└── README.md
```
