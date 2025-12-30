# ====================================
# 文件: feedforward.py
# 前馈神经网络
# ====================================

import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """两层全连接：FFN(x) = ReLU(xW1+b1)W2+b2"""

    def __init__(self, model_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(model_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """x: [batch, seq_len, model_dim]"""
        return self.fc2(
            self.dropout(
                self.relu(
                    self.fc1(x)
                )
            )
        )
