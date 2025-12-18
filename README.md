# PyTorch IMDb Sentiment Binary Classification for Starter

This is a starter text classification project based on PyTorch. It implements sentiment analysis (positive/negative) on the IMDb movie review dataset by building a simple neural network.

This project uses a modular design, **aimed at demonstrating the standard PyTorch development workflow (`custom data loading, model building, training loop`) and the basic structure of "Embedding+MLP"**.

My take on this code: **understand it, commit it to memory, apply it,** and then master it.

### 🚀 1. Environment Setup

This project was developed with **Python 3.12.12**. Please follow the steps below to set up the environment.

```bash
# 1. Make sure pip is up to date
pip install --upgrade pip

# 2. Install all dependencies in one go (specifying the official PyTorch source)
pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

### 📂 2. Project Structure

```text
IMDb-Sentiment-Analysis/
├── config.py           # Configuration center
├── data_loader.py      # Data preparation
├── dataset.py          # Data processing
├── model.py            # Model definition
├── trainer.py          # Training logic
├── main.py             # Main script
├── predict.py          # Inference script
├── requirements.txt    # Dependency list
└── data/               # (Auto-generated) Stores the downloaded dataset
└── models/             # (Auto-generated) Stores the trained model weights
```

### ⚡ 3. Model Training

Running the main script will automatically download the data, process the vocabulary, and start the training:

```bash
python main.py
```

During training, the model with the highest validation accuracy will be automatically saved to `models/best_model.pt`.

### 📊 4. Experimental Results

The results of training the model on the IMDb dataset for 10 epochs are shown below. It can be seen that the model is too simple and suffers from severe **overfitting**.

| Epoch | Train Loss | Test Loss  | Test Accuracy |
| :---: | :--------: | :--------: | :-----------: |
|   1   |   0.6835   |   0.6548   |    61.35%     |
|   2   |   0.5006   |   0.6340   |    66.65%     |
|   3   |   0.1773   |   0.8314   |    66.71%     |
|   4   |   0.0504   |   1.1122   |    67.43%     |
|   5   |   0.0226   |   1.2709   |    67.80%     |
|   6   |   0.0192   |   1.4592   |    67.89%     |
|   7   |   0.0238   |   1.6611   |    68.77%     |
|   8   |   0.0199   |   1.7637   |    68.69%     |
| **9** | **0.0170** | **1.8518** |  **69.28%**   |
|  10   |   0.0098   |   2.0037   |    69.07%     |
# 基于Pytorch的 IMDb 情感二分类 (Starter)

这是一个基于 PyTorch 的入门级文本分类项目。通过构建一个简单的神经网络（Embedding + MLP），实现对 IMDb 电影评论数据集的情感分析（正面/负面）。

该项目采用了模块化设计，**旨在入门学习 PyTorch 标准开发流程**(自定义 data loading、模型构建、训练循环) 和学习"Embedding+MLP"。

**我对这份代码的观点：理解并且记忆，运用而后掌握**

### 🚀 1. 环境配置

实验基于 **Python 3.12.12** 开发。请按照以下步骤配置环境。

```bash
# 1. 确保 pip 是最新的
pip install --upgrade pip

# 2. 一键安装所有依赖 (指定 PyTorch 官方源)
pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

### 📂 2. 项目结构

```text
IMDb-Sentiment-Analysis/
├── config.py           # 配置中心：超参数、路径设置
├── data_loader.py      # 数据准备：下载、解压、读取原始文本
├── dataset.py          # 数据处理：分词、构建词表、Dataset类实现
├── model.py            # 模型定义：EmbeddingBag + 全连接层
├── trainer.py          # 训练逻辑：训练循环、验证、保存模型
├── main.py             # 主程序：串联整个训练流程
├── predict.py          # 推理脚本：加载模型预测新文本
├── requirements.txt    # 依赖列表
└── data/               # (自动生成) 存放下载的数据集
└── models/             # (自动生成) 存放训练好的模型权重
```

### ⚡ 3. 模型训练

运行主程序会自动下载数据、处理词表并开始训练：

```bash
python main.py
```

训练过程中，验证集准确率最高的模型会自动保存到 `models/best_model.pt`。

### 📊 4. 实验结果

模型在 IMDb 数据集上训练了 10 个 Epoch 的结果如下。可以看出模型过于简单，存在严重的**过拟合**现象。

| Epoch | Train Loss | Test Loss | Test Accuracy |
| :---: | :---: | :---: | :---: |
| 1 | 0.6835 | 0.6548 | 61.35% |
| 2 | 0.5006 | 0.6340 | 66.65% |
| 3 | 0.1773 | 0.8314 | 66.71% |
| 4 | 0.0504 | 1.1122 | 67.43% |
| 5 | 0.0226 | 1.2709 | 67.80% |
| 6 | 0.0192 | 1.4592 | 67.89% |
| 7 | 0.0238 | 1.6611 | 68.77% |
| 8 | 0.0199 | 1.7637 | 68.69% |
| **9** | **0.0170** | **1.8518** | **69.28%** |
| 10 | 0.0098 | 2.0037 | 69.07% |
