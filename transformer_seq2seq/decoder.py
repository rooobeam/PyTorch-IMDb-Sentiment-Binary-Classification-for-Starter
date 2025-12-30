# ====================================
# 文件: decoder.py
# Decoder层 + Decoder堆叠
# ====================================

import torch.nn as nn
from attention import MultiHeadAttention
from feedforward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """单个Decoder层：Masked Self-Attention + Cross-Attention + FFN"""

    def __init__(self, model_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(model_dim, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(model_dim, ffn_dim, dropout)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        """
        x: [batch, tgt_len, model_dim] - Decoder输入
        enc_output: [batch, src_len, model_dim] - Encoder输出
        src_mask: [batch, 1, 1, src_len] - Encoder的Pad Mask
        tgt_mask: [batch, 1, tgt_len, tgt_len] - Decoder的Causal Mask

        Transformer 里最容易踩坑的就是 残差相加处用了 +=/+ 后被编译器优化成 in-place。
        把残差里的 `x += sublayer(x)` 全部换成 `x = x + sublayer(x)`，LayerNorm 也先加后归一化，杜绝任何就地改张量，
        即可消除 `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`。
        """
        # Sub-layer 1: Masked Self-Attention（不能看未来信息）
        x_norm = self.norm1(x)
        self_attn_out = self.self_attn(x_norm, x_norm, x_norm, tgt_mask)
        x  = x + self.dropout(self_attn_out)

        # Sub-layer 2: Cross-Attention（Q来自Decoder，K/V来自Encoder）
        x_norm = self.norm2(x)
        cross_attn_out = self.cross_attn(x_norm, enc_out, enc_out, src_mask)
        x = x + self.dropout(cross_attn_out)

        # Sub-layer 3: Feed Forward
        x_norm = self.norm3(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)

        return x


class Decoder(nn.Module):
    """Decoder：N个DecoderLayer堆叠"""

    def __init__(self, num_layers, model_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(model_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x
