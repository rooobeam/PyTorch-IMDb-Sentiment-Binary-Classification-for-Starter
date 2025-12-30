# ====================================
# 文件: transformer.py
# 完整Transformer模型
# ====================================

import torch
import torch.nn as nn
from embeddings import TokenEmbedding
from encoder import Encoder
from decoder import Decoder
from mask import create_padding_mask, create_decoder_mask


class Transformer(nn.Module):
    """完整Transformer：Encoder + Decoder"""

    def __init__(self, config):
        super().__init__()

        # Embedding层
        self.src_embedding = TokenEmbedding(
            config.src_vocab_size, config.max_src_len,
            config.model_dim, config.PAD_ID, config.dropout
        )
        self.tgt_embedding = TokenEmbedding(
            config.tgt_vocab_size, config.max_tgt_len,
            config.model_dim, config.PAD_ID, config.dropout
        )

        # Encoder和Decoder
        self.encoder = Encoder(
            config.num_encoder_layers, config.model_dim,
            config.num_heads, config.ffn_dim, config.dropout
        )
        self.decoder = Decoder(
            config.num_decoder_layers, config.model_dim,
            config.num_heads, config.ffn_dim, config.dropout
        )

        # 输出层（词表概率分布）
        self.output_proj = nn.Linear(config.model_dim, config.tgt_vocab_size)

        self.pad_id = config.PAD_ID

    def forward(self, src, tgt):
        """
        src: [batch, src_len] - 源语言token ids
        tgt: [batch, tgt_len] - 目标语言token ids
        返回: [batch, tgt_len, tgt_vocab_size] - 每个位置的词表概率
        """
        # 生成Mask
        src_mask = create_padding_mask(src, self.pad_id)  # [batch, 1, 1, src_len]
        tgt_mask = create_decoder_mask(tgt, self.pad_id)  # [batch, 1, tgt_len, tgt_len]

        # Encoder
        src_emb = self.src_embedding(src)  # [batch, src_len, model_dim]
        enc_output = self.encoder(src_emb, src_mask)  # [batch, src_len, model_dim]

        # Decoder
        tgt_emb = self.tgt_embedding(tgt)  # [batch, tgt_len, model_dim]
        dec_output = self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)

        # 输出投影 output projection (projection：数学里的“投影”)
        logits = self.output_proj(dec_output)  # [batch, tgt_len, tgt_vocab_size]

        return logits

    def encode(self, src):
        """只运行Encoder（用于推理）"""
        src_mask = create_padding_mask(src, self.pad_id)
        src_emb = self.src_embedding(src)
        return self.encoder(src_emb, src_mask)

    def decode_step(self, tgt, enc_output, src_mask):
        """单步Decoder（用于推理的自回归生成）"""
        tgt_mask = create_decoder_mask(tgt, self.pad_id)
        tgt_emb = self.tgt_embedding(tgt)
        dec_output = self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)
        return self.output_proj(dec_output[:, -1, :])  # 只取最后一个时间步
