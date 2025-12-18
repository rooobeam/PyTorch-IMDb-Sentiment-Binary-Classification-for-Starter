"""
模型模块：定义情感分析模型
"""
import torch
import torch.nn as nn
from config import Config


class SentimentModel(nn.Module):
    """基于 Embedding + 全连接的情感分类模型"""

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = Config.EMBEDDING_DIM,
            hidden_dim: int = Config.HIDDEN_DIM,
            output_dim: int = Config.OUTPUT_DIM,
            seq_len: int = Config.SEQ_LEN,
            dropout1: float = Config.DROPOUT1,
            dropout2: float = Config.DROPOUT2
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim) # [seq_len,] -> [seq_len, embedding_dim]
        self.flatten = nn.Flatten() # [seq_len, embedding_dim] -> [seq_len * embedding_dim,]

        # 全连接层
        self.fc1 = nn.Linear(embedding_dim * seq_len, hidden_dim) # [seq_len * embedding_dim,] -> [hidden_dim,]
        self.dropout1 = nn.Dropout(dropout1)

        self.fc2 = nn.Linear(hidden_dim, output_dim) # [hidden_dim,] -> [output_dim,]
        self.dropout2 = nn.Dropout(dropout2)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len] - 输入的 token indices

        Returns:
            [batch_size] - 预测概率（已经过 sigmoid）
        """
        x = self.embedding(x)  # [B, seq_len, emb_dim]
        x = self.flatten(x)  # [B, seq_len * emb_dim]

        x = self.fc1(x)  # [B, hidden_dim]
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)  # [B, output_dim]
        # dropout2 在这里其实没用到，可能是原作者的失误

        x = torch.sigmoid(x)
        return x.squeeze()  # [B]


class ImprovedSentimentModel(nn.Module):
    """改进版本：使用平均池化而非拉平"""

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = 64,
            hidden_dim: int = 128,
            output_dim: int = 1,
            dropout: float = 0.3
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 使用平均池化，参数量大大减少
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len]
        Returns:
            [batch_size]
        """
        x = self.embedding(x)  # [B, seq_len, emb_dim]
        x = x.mean(dim=1)  # [B, emb_dim] - 平均池化

        x = self.fc1(x)  # [B, hidden_dim]
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)  # [B, output_dim]
        x = torch.sigmoid(x)
        return x.squeeze()


if __name__ == "__main__":
    # 测试模型
    vocab_size = 10000
    model = SentimentModel(vocab_size)

    # 创建假数据
    batch_size = 32
    seq_len = 100
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
