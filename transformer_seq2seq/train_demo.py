# ====================================
# 文件: train_demo.py
# 训练演示（使用真实数据）
# ====================================

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from config import TransformerConfig
from transformer import Transformer
from dataset import load_cornell_dialogues, create_dataloaders


def train_epoch(model, dataloader, optimizer, criterion, device, scheduler):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)

        # Teacher Forcing: 输入tgt[:-1], 预测tgt[1:]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # 前向传播
        logits = model(src, tgt_input)  # [batch, tgt_len-1, vocab_size]

        # 计算损失
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1)
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = model(src, tgt_input)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )

            total_loss += loss.item()

    return total_loss / len(dataloader)


def demo_training():
    """完整训练流程"""

    # 1. 加载数据
    print("=" * 60)
    print("步骤1: 加载Cornell对话数据")
    print("=" * 60)
    train_pairs, val_pairs, vocab = load_cornell_dialogues(
        max_samples=150000,  # 快速实验只用5000对
        max_len=90
    )

    train_loader, val_loader = create_dataloaders(
        train_pairs, val_pairs, vocab, batch_size=96
    )

    # 2. 配置模型
    print("\n" + "=" * 60)
    print("步骤2: 构建Transformer模型")
    print("=" * 60)
    config = TransformerConfig()
    config.src_vocab_size = len(vocab)
    config.tgt_vocab_size = len(vocab)
    config.PAD_ID = vocab.stoi[vocab.pad_token]


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(config).to(device)

    print(f"✅ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✅ 设备: {device}")

    # 3. 训练配置
    optimizer = Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    def warmup_lambda(step):
        warmup_steps = 4000
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_ID)

    # 4. 训练循环
    print("\n" + "=" * 60)
    print("步骤3: 开始训练")
    print("=" * 60)
    num_epochs = 10

    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
        print("-" * 60)

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"✅ Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # 5. 保存模型
    # 在文件末尾添加
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'config': config,
    }, 'transformer_cornell_full.pth')
    print("\n✅ 模型、词表、配置已保存: transformer_cornell_full.pth")


if __name__ == "__main__":
    demo_training()
