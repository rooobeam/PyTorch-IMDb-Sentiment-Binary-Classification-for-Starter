"""
配置文件：集中管理所有超参数和路径配置
"""
import os
from pathlib import Path


class Config:
    # 路径配置
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODEL_DIR = PROJECT_ROOT / "models"

    # 数据集配置
    IMDB_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    DATASET_FILE = DATA_DIR / "aclImdb_v1.tar.gz"
    DATASET_DIR = DATA_DIR / "aclImdb"

    # 文本预处理配置
    SEQ_LEN = 100
    MIN_FREQ = 2  # 词频阈值

    # 模型超参数
    EMBEDDING_DIM = 32
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    DROPOUT1 = 0.2
    DROPOUT2 = 0.35

    # 训练配置
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 6

    # 设备配置
    DEVICE = "cpu"  # 或 "cpu"

    @classmethod
    def ensure_dirs(cls):
        """确保必要的目录存在"""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
