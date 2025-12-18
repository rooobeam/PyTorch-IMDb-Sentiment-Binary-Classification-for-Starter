"""
训练模块：模型训练和评估逻辑
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional

from config import Config


class Trainer:
    """模型训练器"""

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            test_loader: DataLoader,
            device: str = Config.DEVICE,
            learning_rate: float = Config.LEARNING_RATE,
            model_save_path: Optional[Path] = None
    ):
        self.model = model.to(device) # .to(device)!
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        self.criterion = nn.BCELoss() # Binary Cross-Entropy Loss 二分类交叉熵损失：y_pred：[N, *] 任意形状，元素必须在 0~1（已 sigmoid）。y_true：与 y_pred 同形状，元素只能是 0 或 1。
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.model_save_path = model_save_path or Config.MODEL_DIR / "best_model.pt"
        self.best_accuracy = 0.0

        if self.model_save_path.exists():
            self.load_model(self.model_save_path, only_acc = True)



    def train_epoch(self) -> float:
        """训练一个 epoch"""
        self.model.train() # 切换self.model的状态
        epoch_loss = 0.0

        for texts, labels in self.train_loader:
            texts = texts.to(self.device) # .to
            labels = labels.float().to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(texts)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(self.train_loader)

    def evaluate(self, loader: Optional[DataLoader] = None) -> tuple:
        """评估模型"""
        if loader is None:
            loader = self.test_loader

        self.model.eval()
        total_loss = 0.0
        total_correct = 0

        with torch.no_grad(): # 不反向传播，不需要梯度
            for texts, labels in loader:
                texts = texts.to(self.device)
                labels = labels.float().to(self.device)

                outputs = self.model(texts)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() # 这里要脱离计算图

                # 计算准确率
                # outputs >= 0.5   →  bool[4, 1]
                # .float()         →  float[4, 1]
                # preds == labels  →  bool[4, 1]
                # .sum()           →  int64()
                # .item()          →  Python int    把0维张量变为Python原生的标量
                preds = (outputs >= 0.5).float()
                total_correct += (preds == labels).sum().item()

        avg_loss = total_loss / len(loader) # 除以batch数
        accuracy = total_correct / len(loader.dataset) # 除以样本数

        return avg_loss, accuracy



    def train(self, epochs: int = Config.EPOCHS):
        """训练模型"""
        print(f"Training on device: {self.device}")
        print(f"Total epochs: {epochs}\n")

        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch()

            # 验证
            test_loss, test_accuracy = self.evaluate()

            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Test Loss: {test_loss:.4f}, "
                f"Test Accuracy: {test_accuracy:.4f}"
            )

            # 保存最佳模型
            if test_accuracy > self.best_accuracy:
                self.best_accuracy = test_accuracy # 必须先更新self.best_accuracy
                self.save_model()
                print(f"  → Best model saved! (Accuracy: {test_accuracy:.4f})")

        print(f"\nTraining complete! Best accuracy: {self.best_accuracy:.4f}")



    def save_model(self, path: Optional[Path] = None):
        """保存模型"""
        if path is None:
            path = self.model_save_path

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
        }, path)

    def load_model(self, path: Optional[Path] = None, only_acc: bool = False):
        """加载模型"""
        if path is None:
            path = self.model_save_path

        checkpoint = torch.load(path, map_location=self.device) # map_location!!!
        if not only_acc:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model loaded from {path}")
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)

