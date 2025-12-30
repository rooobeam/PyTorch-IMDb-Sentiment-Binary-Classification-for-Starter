# ====================================
# 文件: config.py
# 配置参数
# ====================================

class TransformerConfig:
    # 模型维度
    model_dim = 448
    num_heads = 8
    ffn_dim = 2048
    num_encoder_layers = 4
    num_decoder_layers = 4
    dropout = 0.1

    # 词表
    src_vocab_size = 10000
    tgt_vocab_size = 10000

    # 序列长度
    max_src_len = 100
    max_tgt_len = 100

    # 特殊token
    PAD_ID = 0 # 对齐 Vocabulary类 里的 pad_token
