"""
数据加载模块：负责下载、解压和读取 IMDb 数据集
"""
import re
import urllib.request
import tarfile
from pathlib import Path
from typing import Tuple, List

from config import Config


class IMDbDataLoader:
    """
    IMDb 数据集下载和加载器
    整个类服务于.load_data方法其实
    __init__里如果不传餐则全由config保障参数
    download_and_extract、read_files在load_data里被调用
    remove_html_tags在read_files里被调用
    """

    def __init__(self, data_dir: Path = Config.DATA_DIR):
        self.data_dir = data_dir
        self.dataset_file = Config.DATASET_FILE
        self.dataset_dir = Config.DATASET_DIR
        self.url = Config.IMDB_URL

    def download_and_extract(self):
        """下载并解压数据集"""
        # 下载
        if not self.dataset_file.exists():
            print(f"Downloading dataset from {self.url}...")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self.url, self.dataset_file) # urlretrieve
            print(f"Downloaded to {self.dataset_file}")
        else:
            print(f"Dataset file already exists: {self.dataset_file}")

        # 解压
        if not self.dataset_dir.exists():
            print(f"Extracting dataset to {self.data_dir}...")
            with tarfile.open(self.dataset_file, "r:gz") as tfile:
                tfile.extractall(self.data_dir)
            print("Extraction complete!")
        else:
            print(f"Dataset already extracted: {self.dataset_dir}")

    # 看到 @staticmethod 就立刻明白：“哦，这个方法是独立的，它不会改变或依赖于任何实例的状态”
    @staticmethod
    def remove_html_tags(text: str) -> str:
        """移除 HTML 标签"""
        # r"<[^>]+>" 的匹配过程是：找到一个 <；然后紧接着匹配一个或多个“不是 > 的字符” 即 “[^>]+”；最后找到一个 >
        re_tag = re.compile(r"<[^>]+>")
        return re_tag.sub("", text)

    def read_files(self, filetype: str) -> Tuple[List[int], List[str]]:
        """
        读取指定类型的数据文件（train 或 test）

        Args:
            filetype: "train" 或 "test"

        Returns:
            (labels, texts): 标签列表和文本列表
        """
        path = self.dataset_dir / filetype # 在 self.dataset_dir下
        positive_path = path / "pos"
        negative_path = path / "neg"

        # 收集所有文件路径
        # file_list = (
        #         list(positive_path.glob("*.txt")) +
        #         list(negative_path.glob("*.txt"))
        # )
        #
        # print(f"Reading {filetype} files: {len(file_list)}")
        #
        # # 标签：正面12500个（标签1），负面12500个（标签0） 这里什么情况？？？
        # labels = [1] * 12500 + [0] * 12500
        #
        # # 读取文本
        # texts = []
        # for filepath in file_list:
        #     with open(filepath, encoding="utf8") as f:
        #         text = " ".join(f.readlines())
        #         texts.append(self.remove_html_tags(text))

        pos_file_list = list(positive_path.glob('*.txt'))
        neg_file_list = list(negative_path.glob('*.txt'))

        labels = [1] * len(pos_file_list) + [0] * len(neg_file_list)

        texts = []
        for file_path in pos_file_list + neg_file_list:
            with open(file_path, encoding='utf8') as f:
                text = f.read()
                texts.append(self.remove_html_tags(text))
        return labels, texts

    def load_data(self) -> Tuple[Tuple[List[int], List[str]], Tuple[List[int], List[str]]]:
        """
        加载训练集和测试集

        Returns:
            ((train_labels, train_texts), (test_labels, test_texts))
        """
        self.download_and_extract()

        train_labels, train_texts = self.read_files("train")
        test_labels, test_texts = self.read_files("test")

        return (train_labels, train_texts), (test_labels, test_texts)


if __name__ == "__main__":
    # 测试数据加载
    loader = IMDbDataLoader()
    (train_labels, train_texts), (test_labels, test_texts) = loader.load_data()
    print(f"\nTrain samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    print(f"Sample text: {train_texts[0][:100]}...")
