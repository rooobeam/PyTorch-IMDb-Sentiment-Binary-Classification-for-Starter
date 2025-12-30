# ====================================
# 文件: encoder.py
# Encoder层 + Encoder堆叠
# ====================================

import torch.nn as nn
from attention import MultiHeadAttention
from feedforward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """单个Encoder层：Self-Attention + FFN + 残差连接 + LayerNorm"""

    def __init__(self, model_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(model_dim, ffn_dim, dropout)

        # Layer Normalization（先归一化后相加，Pre-LN架构更稳定）
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(dropout)

        """
        关于 nn.LayerNorm
        作用：把每条样本的最后维 `model_dim` 归一化 → 再**可学习地**缩放/平移回去。  
        公式（逐向量）  
            μ, σ² 在 `model_dim` 上算  
            y = (x − μ) / √(σ² + ε)  
            out = γ ⊙ y + β          # ⊙ 逐元素乘；γ, β 形状 `[model_dim]`  
        广播: `[model_dim]` → 自动复制成 `[B, S, model_dim]`，与输入同形。  
        参数:  γ, β 是 `nn.Parameter`，要显示地传入model_dim，与整个网络一起被梯度更新。
        """

    def forward(self, x, mask):
        """
        x: [batch, src_len, model_dim]
        mask: [batch, 1, 1, src_len]
        """
        # Sub-layer 1: Multi-Head Self-Attention + 残差
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout(attn_out)

        # Sub-layer 2: Feed Forward + 残差
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)

        """
        关于dropout为什么不放在 self_attn/ffn 里
        Transformer 的论文中定义了每个子层（无论是 Attention 还是 FFN）的输出公式都是：LayerNorm(x+Dropout(SubLayer(x))) (post-ln)
        将“残差连接前的 Dropout” 放在 EncoderLayer 里，意味着 所有子层（Attention 和 FFN）都遵循统一的残差处理逻辑：
            Attention 出来 -> Dropout -> 加残差
            FFN 出来 -> Dropout -> 加残差
        """

        return x


class Encoder(nn.Module):
    """Encoder：N个EncoderLayer堆叠"""

    def __init__(self, num_layers, model_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(model_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        """x: [batch, src_len, model_dim]"""
        for layer in self.layers:
            x = layer(x, mask)
        return x
