"""
预测脚本：用训练好的模型对新文本进行情感预测
"""
import torch
from pathlib import Path

from config import Config
from dataset import TextProcessor
from model import SentimentModel
from data_loader import IMDbDataLoader


class SentimentPredictor:
    """情感预测器"""

    def __init__(self, model_path: Path, vocab_texts: list):
        """
        Args:
            model_path: 训练好的模型路径
            vocab_texts: 用于构建词表的文本列表（通常是训练集）
        """
        self.device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")

        # 构建词表（必须和训练时一致）
        self.text_processor = TextProcessor(
            seq_len=Config.SEQ_LEN,
            min_freq=Config.MIN_FREQ
        )
        self.text_processor.build_vocab(vocab_texts)

        # 加载模型
        self.model = SentimentModel(
            vocab_size=len(self.text_processor.vocab),
            seq_len=Config.SEQ_LEN
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded from {model_path}")

    def predict(self, text: str) -> tuple:
        """
        预测单条文本的情感

        Args:
            text: 输入文本

        Returns:
            (label, probability): 预测标签（0/1）和置信度
        """
        # 文本预处理
        text_tensor = self.text_processor.text_to_tensor(text).unsqueeze(0)  # [1, seq_len]
        text_tensor = text_tensor.to(self.device)

        # 预测
        with torch.no_grad():
            prob = self.model(text_tensor).item()

        label = 1 if prob >= 0.5 else 0
        sentiment = "Positive" if label == 1 else "Negative"

        return sentiment, prob

    def predict_batch(self, texts: list) -> list:
        """批量预测"""
        results = []
        for text in texts:
            sentiment, prob = self.predict(text)
            results.append((sentiment, prob))
        return results


def main():
    # 加载训练数据以构建词表
    print("Loading training data for vocabulary...")
    data_loader = IMDbDataLoader()
    (train_labels, train_texts), _ = data_loader.load_data()

    # 创建预测器
    model_path = Config.MODEL_DIR / "best_model.pt"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first by running main.py")
        return

    predictor = SentimentPredictor(model_path, train_texts)

    # 测试样例
    test_texts = [
        "This movie is absolutely amazing! I loved every minute of it.",
        "Terrible waste of time. The worst movie I've ever seen.",
        "It was okay, nothing special but not bad either.",
        "Brilliant performance by the actors. Highly recommended!",
        "Boring and predictable plot. Very disappointed."
    ]

    print("\n" + "=" * 60)
    print("Sentiment Prediction Results")
    print("=" * 60)

    for text in test_texts:
        sentiment, prob = predictor.predict(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment} (confidence: {prob:.4f})")


if __name__ == "__main__":
    main()
