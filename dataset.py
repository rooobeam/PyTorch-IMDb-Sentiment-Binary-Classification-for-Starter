"""
数据集模块：定义 PyTorch Dataset，集成文本预处理
"""
import torch
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator, Vocab
from typing import List, Optional

from config import Config


class TextProcessor:
    """文本处理器：分词、构建词表、文本转张量"""

    def __init__(self, seq_len: int = Config.SEQ_LEN, min_freq: int = Config.MIN_FREQ):
        self.seq_len = seq_len
        self.min_freq = min_freq
        self.vocab: Optional[Vocab] = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """简单的分词：转小写并按空格分割"""
        return text.strip().lower().split() # lower() 词表一致性

    def build_vocab(self, texts: List[str]):
        """从文本列表构建词表"""

        def yield_tokens():
            for text in texts:
                yield self.tokenize(text)

        self.vocab = build_vocab_from_iterator(
            yield_tokens(),
            specials=["<unk>", "<pad>"],
            min_freq=self.min_freq
        )
        self.vocab.set_default_index(self.vocab["<unk>"]) # 设置默认！！！
        print(f"Vocabulary size: {len(self.vocab)}")
        return self.vocab

    def text_to_tensor(self, text: str) -> torch.Tensor:
        """将文本转换为张量（tokenize + index + truncate/pad）"""
        if self.vocab is None:
            raise ValueError("Vocabulary not built! Call build_vocab() first.")

        # 分词并转为索引
        tokens = self.tokenize(text)
        indexed = [self.vocab[token] for token in tokens]

        # 截断或填充到固定长度
        if len(indexed) >= self.seq_len:
            padded = indexed[:self.seq_len]
        else:
            pad_idx = self.vocab["<pad>"]
            padded = indexed + [pad_idx] * (self.seq_len - len(indexed))

        return torch.tensor(padded, dtype=torch.long)


class SentimentDataset(Dataset):
    """情感分析数据集"""

    def __init__(
            self,
            texts: List[str],
            labels: List[int],
            text_processor: TextProcessor,
            transform_on_the_fly: bool = False
    ):
        """
        Args:
            texts: 原始文本列表
            labels: 标签列表（0或1）
            text_processor: 文本处理器
            transform_on_the_fly: 是否在 __getitem__ 时才转换文本（内存换时间）
        """
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.text_processor = text_processor
        self.transform_on_the_fly = transform_on_the_fly

        if transform_on_the_fly:
            # 保存原始文本，在使用时才转换（节省内存）
            self.texts = texts
        else:
            # 预先转换所有文本为张量（节省时间）
            print("Converting all texts to tensors...")
            self.texts = torch.stack([
                text_processor.text_to_tensor(text) for text in texts
            ])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        if self.transform_on_the_fly:
            text_tensor = self.text_processor.text_to_tensor(self.texts[idx])
        else:
            text_tensor = self.texts[idx]

        return text_tensor, self.labels[idx]

if __name__ == "__main__":
    # 测试代码
    from data_loader import IMDbDataLoader

    # 加载数据
    loader = IMDbDataLoader()
    (train_labels, train_texts), _ = loader.load_data()

    # 构建词表
    processor = TextProcessor(seq_len=100)
    processor.build_vocab(train_texts[:1000])  # 用部分数据测试

    # 创建数据集
    dataset = SentimentDataset(
        train_texts[:100],
        train_labels[:100],
        processor,
        transform_on_the_fly=False
    )

    print(f"Dataset size: {len(dataset)}")
    text, label = dataset[0]
    print(f"Sample shape: {text.shape}, label: {label}")
