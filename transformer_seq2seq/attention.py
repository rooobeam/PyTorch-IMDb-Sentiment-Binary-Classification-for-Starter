# ====================================
# 文件: attention.py
# 多头注意力机制
# ====================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """多头注意力：Attention(Q,K,V) = softmax(QK^T/√d_k)V"""

    def __init__(self, model_dim, num_heads, dropout=0.1):
        super().__init__()
        assert model_dim % num_heads == 0

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads  # 每个头的维度
        self.scale = self.head_dim ** -0.5  # 1/√d_k

        # 4个线性层：Q、K、V、输出
        self.W_Q = nn.Linear(model_dim, model_dim)
        self.W_K = nn.Linear(model_dim, model_dim)
        self.W_V = nn.Linear(model_dim, model_dim)
        self.W_O = nn.Linear(model_dim, model_dim) # (remember 还有一个 W_O !!!)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        query/key/value: [batch, seq_len, model_dim]
        mask: [batch, 1, 1, seq_len] (broadcast到多头)
              True表示需要mask的位置(设为-inf)
        """
        batch_size = query.size(0)

        # --- 线性投影 + 拆分多头
        # [batch, seq_len, model_dim] -> [batch, seq_len, num_heads, head_dim]
        Q = self.W_Q(query).reshape(batch_size, -1, self.num_heads, self.head_dim)
        K = self.W_K(key).reshape(batch_size, -1, self.num_heads, self.head_dim)
        V = self.W_V(value).reshape(batch_size, -1, self.num_heads, self.head_dim)

        # 转置为 [batch, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # --- 计算注意力分数：QK^T/√d_k
        # [batch, num_heads, q_len, head_dim] @ [batch, num_heads, head_dim, k_len]
        # -> [batch, num_heads, q_len, k_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (remember * self.scale !!!)

        # 应用mask（pad mask （+ 未来信息mask））
        if mask is not None:
            scores = scores.masked_fill(mask == True, float('-inf'))
        """
        广播机制
        维度从右向左，只要每一步都满足“相等 或 其中一个是 1”，广播就能成功
        [batch, num_heads, q_len, k_len]
                       [batch,1,1,k_len] 能广播
                           [batch,k_len] batch!=q_len，就算相等也是错的，不能广播
        """

        # --- 转为权重 Softmax + Dropout
        attn_weights = F.softmax(scores, dim=-1)  # [batch, num_heads, q_len, k_len]  (remember dim=-1 !!!)
        attn_weights = self.dropout(attn_weights)

        # --- 对 V 加权求和：Attention * V
        # [batch, num_heads, q_len, k_len] @ [batch, num_heads, k_len, head_dim]
        # -> [batch, num_heads, q_len, head_dim]
        attn_output = torch.matmul(attn_weights, V)

        # --- 合并多头
        # [batch, num_heads, q_len, head_dim] -> [batch, q_len, model_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, -1, self.model_dim)

        # 最终线性层
        return self.W_O(attn_output)