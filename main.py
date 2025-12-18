"""
主程序：整合所有模块，完成完整的训练流程
"""
import torch
from torch.utils.data import DataLoader

from config import Config
from data_loader import IMDbDataLoader
from dataset import TextProcessor, SentimentDataset
from model import SentimentModel, ImprovedSentimentModel
from trainer import Trainer


def main():
    # 确保目录存在
    Config.ensure_dirs()

    # ========== 1. 数据加载 ==========
    print("=" * 50)
    print("Step 1: Loading data...")
    print("=" * 50)

    data_loader = IMDbDataLoader()
    (train_labels, train_texts), (test_labels, test_texts) = data_loader.load_data()

    # ========== 2. 构建词表 ==========
    print("\n" + "=" * 50)
    print("Step 2: Building vocabulary...")
    print("=" * 50)

    text_processor = TextProcessor()
    vocab = text_processor.build_vocab(train_texts)

    # ========== 3. 创建数据集和数据加载器 ==========
    print("\n" + "=" * 50)
    print("Step 3: Creating datasets...")
    print("=" * 50)

    train_dataset = SentimentDataset(
        train_texts,
        train_labels,
        text_processor,
        transform_on_the_fly=False  # 预先转换，加快训练
    )

    test_dataset = SentimentDataset(
        test_texts,
        test_labels,
        text_processor,
        transform_on_the_fly=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        pin_memory=True if Config.DEVICE == "cuda" else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        pin_memory=True if Config.DEVICE == "cuda" else False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # ========== 4. 创建模型 ==========
    print("\n" + "=" * 50)
    print("Step 4: Creating model...")
    print("=" * 50)

    # 选择模型
    model = SentimentModel(
        vocab_size=len(vocab),
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        output_dim=Config.OUTPUT_DIM,
        seq_len=Config.SEQ_LEN
    )

    # 或使用改进版模型（参数更少，训练更快）
    # model = ImprovedSentimentModel(
    #     vocab_size=len(vocab),
    #     embedding_dim=64,
    #     hidden_dim=128
    # )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ========== 5. 训练模型 ==========
    print("\n" + "=" * 50)
    print("Step 5: Training model...")
    print("=" * 50)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=Config.DEVICE,
        learning_rate=Config.LEARNING_RATE
    )

    trainer.train(epochs=Config.EPOCHS)

    # ========== 6. 最终评估 ==========
    print("\n" + "=" * 50)
    print("Step 6: Final evaluation...")
    print("=" * 50)

    # 加载最佳模型
    trainer.load_model()
    test_loss, test_accuracy = trainer.evaluate()

    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")

def main():
    Config.ensure_dirs()

    # 1 加载IMDB数据
    data_loader = IMDbDataLoader()
    (train_labels, train_texts), (test_labels, test_texts) = data_loader.load_data()

    # 2 生成词典
    text_processor = TextProcessor()
    vocab = text_processor.build_vocab(train_texts)

    # 3 生成dataset和dataloader

    train_dataset = SentimentDataset(
        texts = train_texts,
        labels = train_labels,
        text_processor = text_processor,
        transform_on_the_fly = False,
    )

    test_dataset = SentimentDataset(
        texts=test_texts,
        labels=test_labels,
        text_processor=text_processor,
        transform_on_the_fly=False,
    )

    train_loader = DataLoader(
        dataset = train_dataset,
        shuffle = True,
        batch_size = Config.BATCH_SIZE,
        pin_memory = True if Config.DEVICE == 'cuda' else False
    )

    test_loader = DataLoader(
        dataset = test_dataset,
        shuffle=False, # 测试集不必打乱
        batch_size=Config.BATCH_SIZE,
        pin_memory=True if Config.DEVICE == 'cuda' else False
    )

    # 4 创建模型

    model = SentimentModel(
        vocab_size = len(vocab),
        seq_len = Config.SEQ_LEN,
        embedding_dim = Config.EMBEDDING_DIM,
        hidden_dim =  Config.HIDDEN_DIM,
        output_dim = Config.OUTPUT_DIM
    )

    # 5 训练模型

    trainer = Trainer(
        model = model,
        train_loader = train_loader,
        test_loader = test_loader,
        device = Config.DEVICE,
        learning_rate = Config.LEARNING_RATE
    )
    trainer.train(epochs = Config.EPOCHS)

    # 6 模型评估

    trainer.load_model()
    loss, acc = trainer.evaluate()
    print(f"loss: {loss}, acc: {acc}")
if __name__ == "__main__":
    main()
