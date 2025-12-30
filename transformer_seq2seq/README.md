# PyTorch Transformer Seq2Seq Dialogue Generation

This is a Transformer-based sequence-to-sequence (seq2seq) dialogue generation project. It builds a complete Transformer model and trains it on the Cornell Movie Dialogs dataset to generate conversational responses.

This project uses a modular design, **aimed at mastering the practical nuts-and-bolts of Transformers and more advanced PyTorch usage.**.

### 🚀 1. Environment Setup

This project was developed with **Python 3.12.12**. Please follow the steps below to set up the environment.

```bash
# 1. Make sure pip is up to date
pip install --upgrade pip

# 2. Install all dependencies
pip install --no-cache-dir -r requirements.txt
```

### 📂 2. Project Structure

```text
transformer_seq2seq/
├── attention.py        # Multi-head attention mechanism
├── config.py           # Configuration center
├── dataset.py          # Dataset loading and processing
├── decoder.py          # Decoder layer and stack
├── embeddings.py       # Token embedding + positional encoding
├── encoder.py          # Encoder layer and stack
├── feedforward.py      # Position-wise feedforward network
├── inference_demo.py   # Interactive inference demo
├── mask.py             # Mask generation tools
├── requirements.txt    # Dependency list
├── train_demo.py       # Training demo
└── transformer.py      # Complete Transformer model
```

### ⚡ 3. Model Training

Running the training script will automatically download the Cornell Movie Dialogs dataset, process the vocabulary, and start the training:

```bash
python train_demo.py
```

During training:
- The model uses teacher forcing for efficient training
- Gradient clipping is applied to prevent gradient explosion
- A warmup learning rate scheduler is used
- The trained model, vocabulary, and configuration will be saved to `transformer_cornell_full.pth`

### 💬 4. Dialogue Generation

After training, you can use the interactive inference demo to test the dialogue generation:

```bash
python inference_demo.py
```

Enter your message and the bot will generate a response based on the trained model.

### 📋 5. Core Implementation Details

#### Transformer Architecture  
The model implements the full Transformer architecture:  
- **Embedding Layer**: combines word embeddings with sinusoidal positional encodings  
- **Multi-Head Attention**: parallel attention heads capturing different contextual aspects  
- **Encoder**: stacked self-attention layers + feed-forward networks  
- **Decoder**: stacked masked self-attention + cross-attention + feed-forward networks  
- **Output Layer**: projects decoder outputs to vocabulary size  

```
transformer.py |<- embeddings.py
               |<- encoder.py  <- attention.py
               |<- decoder.py  <- attention.py
               |<- mask.py
```

#### Training Pipeline  
1. **Data Preparation**: auto-download & preprocess Cornell Movie Dialogs, build vocabulary, create DataLoader  
2. **Model Setup**: config_init, model_init, criterion, optimizer, ...  
3. **Training Loop**: full training and evaluation routine  
4. **Model Checkpointing**: save the complete model, vocabulary, and configuration  

```
train_demo.py |<- dataset.py
              |<- config.py
              |<- transformer.py
```

#### Inference Pipeline  
After training, the chat-generation flow is:  
1. **Model Loading**: load the trained model, vocabulary, and config from `transformer_cornell_full.pth`  
2. **User Input, Tokenization, Numericalization**: convert user input → tokens → token IDs  
3. **greedy_decode**  
4. **Text Conversion**: map generated token IDs back to text and print

# 基于Pytorch的Transformer Seq2Seq对话生成

这是一个基于Transformer的序列到序列（Seq2Seq）对话生成项目。它构建了完整的Transformer模型，并在Cornell电影对话数据集上进行训练，以生成对话响应。

该项目采用模块化设计，**旨在学习Transformer的实践细节和Pytorch的更多用法**。

### 🚀 1. 环境配置

实验基于 **Python 3.12.12** 开发。请按照以下步骤配置环境。

```bash
# 1. 确保 pip 是最新的
pip install --upgrade pip

# 2. 安装所有依赖
pip install --no-cache-dir -r requirements.txt
```

### 📂 2. 项目结构

```text
transformer_seq2seq/
├── attention.py        # 多头注意力机制
├── config.py           # 配置中心
├── dataset.py          # 数据集加载与处理
├── decoder.py          # Decoder层与堆叠
├── embeddings.py       # 词嵌入 + 位置编码
├── encoder.py          # Encoder层与堆叠
├── feedforward.py      # 位置前馈网络
├── inference_demo.py   # 交互式推理演示
├── mask.py             # Mask生成工具
├── requirements.txt    # 依赖列表
├── train_demo.py       # 训练演示
└── transformer.py      # 完整Transformer模型
```

### ⚡ 3. 模型训练

运行训练脚本将自动下载Cornell电影对话数据集、处理词表并开始训练：

```bash
python train_demo.py
```

训练过程中：
- 模型使用teacher forcing进行高效训练
- 应用梯度裁剪防止梯度爆炸
- 使用warmup学习率调度器
- 训练完成后，模型、词表和配置将保存到 `transformer_cornell_full.pth`

### 💬 4. 对话生成

训练完成后，你可以使用交互式推理演示测试对话生成：

```bash
python inference_demo.py
```

输入你的消息，机器人将根据训练好的模型生成响应。

### 📋 5. 核心实现细节

#### Transformer架构
模型实现了完整的Transformer架构：
- **嵌入层**：结合词嵌入与正弦位置编码
- **多头注意力**：并行注意力机制捕获上下文的不同方面
- **编码器**：自注意力层堆叠 + 前馈网络
- **解码器**：掩码自注意力 + 交叉注意力 + 前馈网络堆叠
- **输出层**：将解码器输出投影到词表大小

```
transformer.py |<-embeddings.py
			  |<-encoder.py <- attention.py
			  |<-decoder.py <- attention.py
			  |<-mask.py
```

#### 训练流程
1. **数据准备**：自动下载、处理Cornell成原始数据和构建词表、构建dataloader
2. **模型配置和训练准备**：config_init、model_init、criterion、optimizer...
4. **训练循环**：实现完整的训练和评估流程
5. **模型保存**：保存完整的模型、词表和配置

```
train_demo.py | <- dataset.py
			 | <- config.py
			 | <- transfomer.py
```

#### 推理流程
训练完成后，生成对话的流程如下：
1. **模型加载**：从保存的检查点 `transformer_cornell_full.pth` 中加载训练好的模型、词表和配置
2. **用户输入、分词、数值化**：用户输入转换为tokens 转换为token IDs
3. **greedy_decode** 
4. **文本转换**：将生成的token IDs转换回文本并输出
