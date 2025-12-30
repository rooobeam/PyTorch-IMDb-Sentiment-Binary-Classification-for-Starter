# ====================================
# 文件: embeddings.py
# 词嵌入 + 位置编码
# ====================================

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """正弦位置编码：PE(pos, 2i) = sin(pos/10000^(2i/d))
                     PE(pos, 2i+1) = cos(pos/10000^(2i/d))
    """

    def __init__(self, max_len, model_dim):
        super().__init__()
        # 构造位置编码表 [max_len, model_dim]
        pos = torch.arange(max_len).reshape(-1, 1)  # [max_len, 1]
        i = torch.arange(0, model_dim, 2).reshape(1, -1)  # [1, model_dim//2]
        denom = torch.pow(10000, i / model_dim)  # [1, model_dim//2]  denominator/dɪˈnɑːmɪneɪtər/ 分母

        pe_table = torch.zeros(max_len, model_dim)
        pe_table[:, 0::2] = torch.sin(pos / denom)  # 偶数列
        pe_table[:, 1::2] = torch.cos(pos / denom)  # 奇数列

        self.register_buffer('pe_table', pe_table)  # 不参与梯度更新

        """
        def register_buffer(self, name, tensor):
            self._buffers[name] = tensor # 1. 把它存进内部的一个字典（_buffers）里，为了以后保存模型、切GPU用，
            setattr(self, name, tensor) # 2. 使用内置函数 setattr 动态生成属性，后面能用 self.pe_table 访问它的原因
            
        nn.Module 内部，只有 _parameters  → 可训练参数（nn.Parameter）
                            _buffers     → 不可训练但需持久化的缓存
        这两本“注册表”会被 state_dict() 自动收集。
        """

    def forward(self, token_emb):
        """token_emb: [batch, seq_len, model_dim]"""
        seq_len = token_emb.shape[1]
        return token_emb + self.pe_table[:seq_len, :]  # 广播相加


class TokenEmbedding(nn.Module):
    """词嵌入 + 位置编码 + Dropout"""

    def __init__(self, vocab_size, max_len, model_dim, padding_idx=0,dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, model_dim, padding_idx) # vocab_size是确保，无论哪个词来，都能embedding
        self.pos_enc = PositionalEncoding(max_len, model_dim) # max_len是确保，无论哪个batch来，都能大于其seq_len
        self.dropout = nn.Dropout(dropout)
        # self.scale = math.sqrt(model_dim)  # 论文中的缩放因子
        """
        移除 scale (sqrt(d))，因采用默认初始化 N(0,1)：
        1. Emb 模长：E[x^2] = Var(x) + E[x]^2 = 1，故 ||Emb|| ≈ sqrt(∑E[x^2]) = sqrt(d_model)。
        2. PE 模长：Sinusoid 均方值为 0.5 (sin^2+cos^2=1)，故 ||PE|| ≈ sqrt(d_model * 0.5)。
        3. 结论：||Emb|| ≈ 1.414 * ||PE||，二者天然处于同一数量级，无需额外缩放。
        标准做法（Paper / Xavier Init）：大家通常会手动重置 Embedding 的初始化，使其方差为 1/d_model （Xavier/Glorot 方式）。
        理论上这让优化器起步更平滑，搜索空间更“细腻”，但懒得改self.token_emb, 优化器自适应能力很强，只要数值没有爆炸（NaN），通常都能把梯度拉回来。
        
        nn.Embedding(..., padding_idx):
        告诉 Embedding 层：哪个id是 填充符，遇到它时直接把向量置成全 0，且该向量在反向传播时不参与梯度更新。
        正向传播：output[…, padding_idx, …] == 0。
        反向传播：对 padding_idx 这一行的梯度直接丢弃，因此该行参数永远不会被更新，始终保持初始值（默认也是 0）。
        """

    def forward(self, token_ids):
        """token_ids: [batch, seq_len]"""
        emb = self.token_emb(token_ids)  # [batch, seq_len, model_dim]
        emb = self.pos_enc(emb)
        return self.dropout(emb) # 就像让模型做填空题，即使输入的某些特征丢了，模型依然要靠上下文把后续的任务完成，提高模型的泛化能力。
