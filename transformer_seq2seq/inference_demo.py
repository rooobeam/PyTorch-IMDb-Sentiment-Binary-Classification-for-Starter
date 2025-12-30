# ====================================
# 文件: inference_demo.py
# 推理演示（真实对话生成）
# ====================================

import torch
from config import TransformerConfig
from transformer import Transformer
from dataset import simple_tokenize


def greedy_decode(model, src, vocab, max_len=20):
    """
    贪心解码：每次选概率最大的词
    src: [1, src_len] - 单个句子
    """
    model.eval()
    device = next(model.parameters()).device
    src = src.to(device)

    bos_id = vocab.stoi["<BOS>"]
    eos_id = vocab.stoi["<EOS>"]

    # Encoder一次性编码
    enc_output = model.encode(src)
    src_mask = (src == model.pad_id).unsqueeze(1).unsqueeze(2)

    # Decoder自回归生成
    tgt = torch.tensor([[bos_id]], device=device)  # [1, 1]

    for _ in range(max_len):
        # 预测下一个词
        logits = model.decode_step(tgt, enc_output, src_mask)  # [1, vocab_size]
        next_token = logits.argmax(dim=-1, keepdim=True)  # [1, 1]

        # 拼接到序列
        tgt = torch.cat([tgt, next_token], dim=1)

        # 遇到结束符则停止
        if next_token.item() == eos_id:
            break

    return tgt.squeeze(0)  # [tgt_len]


def chat_demo():
    """交互式对话演示"""

    # 加载模型和词表
    print("🤖 加载模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('transformer_cornell_full.pth', map_location=device, weights_only = False)
    vocab = checkpoint['vocab']
    config = checkpoint['config']


    print(f"device: {device}")
    model = Transformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.pad_id = config.PAD_ID
    model.eval()

    print("✅ 模型加载完成！\n")
    print("=" * 60)
    print("💬 开始对话（输入 'quit' 退出）")
    print("=" * 60)
    print([name for name, _ in model.named_children()])
    print(f"模型输出层维度: {model.output_proj.out_features}")
    print(f"真实词表大小: {len(vocab)}")
    while True:
        # 用户输入
        user_input = input("\n你: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 再见！")
            break

        # 分词
        tokens = simple_tokenize(user_input)
        if not tokens:
            print("Bot: ...")
            continue

        # 转为ID
        src_ids = [vocab.stoi["<BOS>"]] + vocab.numericalize(tokens) + [vocab.stoi["<EOS>"]]
        src = torch.tensor([src_ids])  # [1, src_len]

        # 生成回复
        with torch.no_grad():
            output_ids = greedy_decode(model, src, vocab, max_len=20)

        # 解码为文本
        output_tokens = [vocab.itos[idx.item()] for idx in output_ids]
        output_tokens = [t for t in output_tokens if t not in ["<BOS>", "<EOS>", "<PAD>"]]

        response = " ".join(output_tokens)
        print(f"Bot: {response}")


if __name__ == "__main__":
    chat_demo()
