# ====================================
# 文件: mask.py
# Mask生成工具
# ====================================

import torch


def create_padding_mask(seq, pad_id=0):
    """
    生成Padding Mask：PAD位置为True
    seq: [batch, seq_len]
    返回: [batch, 1, 1, seq_len]（广播到多头和所有q）
    所有q和这些位置的k的得分 是 要置为负无穷
    """
    mask = (seq == pad_id).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
    return mask


def create_causal_mask(seq_len):
    """
    生成Causal Mask（下三角矩阵）：防止看到未来信息  (causal /ˈkɔzəl/ casual /ˈkæʒuəl/)
    返回: [1, 1, seq_len, seq_len]

    示例 (seq_len=4):
    [[False, True,  True,  True ],
     [False, False, True,  True ],
     [False, False, False, True ],
     [False, False, False, False]]

    `torch.triu(input, diagonal=0)` ：
        把输入矩阵的 下三角 置0。  triangular upper
        参数 diagonal 控制对角线的偏移量。
        diagonal=0：保留主对角线及以上。
        diagonal=1：保留主对角线往上偏移一格之后的部分（即不包含主对角线）。
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]



def create_decoder_mask(tgt_seq, pad_id=0):
    """
    结合Padding Mask和Causal Mask
    tgt_seq: [batch, tgt_len]
    返回: [batch, 1, tgt_len, tgt_len]
    """
    batch_size, tgt_len = tgt_seq.shape

    # Padding Mask: [batch, 1, 1, tgt_len]
    pad_mask = create_padding_mask(tgt_seq, pad_id)

    # Causal Mask: [1, 1, tgt_len, tgt_len]
    causal_mask = create_causal_mask(tgt_len).to(tgt_seq.device)

    # 合并：任一为True则为True
    # 广播： [batch, 1, 1, tgt_len] | [1, 1, tgt_len, tgt_len]
    # -> [batch, 1, tgt_len, tgt_len]
    return pad_mask | causal_mask
