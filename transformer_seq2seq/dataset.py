# ====================================
# 文件: dataset.py
# 数据集加载（使用Cornell电影对话数据集）
# ====================================

import torch
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
import urllib.request
import zipfile
from pathlib import Path
from typing import List

class Vocabulary:
    """词表构建工具"""

    def __init__(self, freq_threshold=2):
        # 特殊token
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        self.freq_threshold = freq_threshold
        self.itos = {0: self.pad_token, 1: self.bos_token,
                     2: self.eos_token, 3: self.unk_token}
        self.stoi = {v: k for k, v in self.itos.items()}

    def build_vocabulary(self, sentence_list):
        """从句子列表构建词表"""
        counter = Counter()
        for sentence in sentence_list:
            counter.update(sentence)

        #  string to index
        idx = len(self.stoi)
        for word, freq in counter.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text:List[str]):
        """文本转token ids"""
        unk_idx = self.stoi[self.unk_token]
        return [self.stoi.get(token, unk_idx) for token in text]

    def __len__(self):
        return len(self.stoi)
class Vocabulary:
    def __init__(self,freq_threshold=2):
        self.pad_token = '<PAD>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'
        self.freq_threshold=freq_threshold
        self.itos = {0: self.pad_token, 1: self.bos_token,
                     2: self.eos_token, 3: self.unk_token}
        self.stoi = {v: k for k, v in self.itos.items()}

    def build_vocab(self, all_sentences):
        counter = Counter()
        for sentence in all_sentences:
            counter.update(sentence)
        index = len(self.stoi)
        for word, freq in counter:
            if freq>=self.freq_threshold:
                self.itos[index] = word
                self.


    def numericalize(self, text):

    def __len__(self):
        return

class DialogueDataset(Dataset):
    """对话数据集"""

    def __init__(self, data_pairs, vocab):
        self.data_pairs = data_pairs
        self.vocab = vocab

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data_pairs[idx]

        # 转为id（添加BOS/EOS）
        src_ids = [self.vocab.stoi["<BOS>"]] + \
                  self.vocab.numericalize(src_text) + \
                  [self.vocab.stoi["<EOS>"]]

        tgt_ids = [self.vocab.stoi["<BOS>"]] + \
                  self.vocab.numericalize(tgt_text) + \
                  [self.vocab.stoi["<EOS>"]]

        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def simple_tokenize(text):
    """
    简单分词：小写+空格分割
    “把不在白名单里的字符统统删掉”。
    白名单里有哪些？ a–z 0–9 \s（空格、制表符、换行等所有空白字符） ?!.,‘
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\?\!\.\,']", "", text)  # 保留基本标点
    tokens = text.split()
    return tokens


def collate_fn(batch, pad_idx=0):
    """动态Padding到batch内最大长度"""
    src_batch, tgt_batch = zip(*batch)

    # 获取batch内最大长度
    src_max_len = max(len(s) for s in src_batch)
    tgt_max_len = max(len(t) for t in tgt_batch)

    # Padding
    src_padded = torch.stack([
        torch.cat([s, torch.tensor([pad_idx] * (src_max_len - len(s)), dtype=torch.int32)])
        for s in src_batch
    ])

    tgt_padded = torch.stack([
        torch.cat([t, torch.tensor([pad_idx] * (tgt_max_len - len(t)), dtype=torch.int32)])
        for t in tgt_batch
    ])

    return src_padded, tgt_padded


def download_cornell_data(data_dir: Path = Path('./data')):
    """下载Cornell Movie Dialogs数据集"""
    data_dir.mkdir(parents = True, exist_ok = True)

    zip_file = data_dir / Path('cornell_movie_dialogs.zip') # 文件用path
    extract_dir = data_dir / Path('cornell movie-dialogs corpus') # 目录用dir
    url = 'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'

    if not zip_file.exists():
        print(f"📥 下载Cornell数据集...")
        urllib.request.urlretrieve(url, zip_file)
        print("✅ 下载完成")

    # 解压
    if not extract_dir.exists():
        print("📦 解压文件...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("✅ 解压完成")

    return extract_dir


# def load_cornell_dialogues(data_dir: Path = Path('./data'), max_samples=5000, max_len=15):
#     """
#     加载Cornell电影对话数据集
#
#     参数：
#     - max_samples: 最多加载多少对对话（默认5000，适合快速实验）
#     - max_len: 过滤掉超过此长度的句子（避免过长句子）
#     """
#     # 下载数据
#     corpus_dir = download_cornell_data(data_dir)
#
#     # 文件路径
#     lines_file = corpus_dir / Path('movie_lines.txt')
#     conv_file = corpus_dir / Path('movie_conversations.txt')
#     assert(lines_file.exists() and conv_file.exists())
#
#     # 解析台词 id2line
#     print("📖 解析电影台词...")
#     id2line = {}
#     with open(lines_file, 'r', encoding='iso-8859-1') as f:
#         for line in f:
#             parts = line.strip().split(' +++$+++ ')
#             if len(parts) == 5:
#                 line_id = parts[0]
#                 text = parts[-1]
#                 id2line[line_id] = text
#
#     # 构造对话 dialogues List[Tuple]
#     print("🔗 构建对话对...")
#     dialogues = []
#     with open(conv_file, 'r', encoding='iso-8859-1') as f:
#         for line in f:
#             parts = line.strip().split(' +++$+++ ')
#             if len(parts) == 4:
#                 line_ids = eval(parts[-1])  # ['L1', 'L2', 'L3']
#                 # 构建Q-A对
#                 for i in range(len(line_ids) - 1):
#                     q_id = line_ids[i]
#                     a_id = line_ids[i + 1]
#                     if q_id in id2line and a_id in id2line:
#                         question = id2line[q_id]
#                         answer = id2line[a_id]
#                         dialogues.append((question, answer))
#
#     print(f"📊 原始对话数量: {len(dialogues)}")
#
#     # 分词 + 过滤长句
#     print("✂️  分词并过滤...")
#     filtered_dialogues = []
#     all_sentences = []
#
#     for q, a in dialogues:
#         q_tokens = simple_tokenize(q)
#         a_tokens = simple_tokenize(a)
#
#         # 过滤：长度合理 + 非空
#         if (1 <= len(q_tokens) <= max_len and
#                 1 <= len(a_tokens) <= max_len):
#             filtered_dialogues.append((q_tokens, a_tokens))
#             all_sentences.extend([q_tokens, a_tokens])
#
#             if len(filtered_dialogues) >= max_samples:
#                 break
#
#     print(f"✅ 过滤后对话数量: {len(filtered_dialogues)}")
#
#     # 构建词表
#     print("📚 构建词表...")
#     vocab = Vocabulary(freq_threshold=2)
#     vocab.build_vocabulary(all_sentences)
#     print(f"📖 词表大小: {len(vocab)}")
#
#     # 数据集划分
#     train_size = int(0.9 * len(filtered_dialogues))
#     train_pairs = filtered_dialogues[:train_size]
#     val_pairs = filtered_dialogues[train_size:]
#
#     print(f"🎯 训练集: {len(train_pairs)} | 验证集: {len(val_pairs)}")
#
#     return train_pairs, val_pairs, vocab

def load_cornell_dialogues(data_dir:Path = Path('./data'), max_samples=5000, max_len=15):

    # 数据下载
    corpus_dir = download_cornell_data(data_dir)

    # 文件路径
    lines_file = corpus_dir / Path('movie_lines.txt')
    conv_file = corpus_dir / Path('movie_conversations.txt')
    assert(lines_file.exists() and conv_file.exists())

    # 解析lines文件 -> id2lines
    id2lines = dict()
    with open(lines_file, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.strip().split(' +++$+++ ')
            if len(parts)==5:
                line_id = parts[0]
                text = parts[-1]
                id2lines[line_id]=text

    # 解析conv文件 -> dialogues
    dialogues = []
    with open(conv_file, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.strip().split(' +++$+++ ')
            if len(parts)==4:
                line_ids = eval(parts[-1])
                for i in range(len(line_ids)-1):
                    q_id, a_id = line_ids[i], line_ids[i+1]
                    if a_id in id2lines and q_id in id2lines:
                        dialogues.append((id2lines[a_id], id2lines[q_id]))

    print(f"原始对话数：{len(dialogues)}")

    # 根据max_len max_samples过滤
    filtered_pairs = []
    all_sentences = []
    for q, a in dialogues:
        q_tokens = simple_tokenize(a)
        a_tokens = simple_tokenize(a)
        if (1<=len(a_tokens)<=max_len
                and 1<=len(q_tokens)<=max_len):
            filtered_pairs.append((q_tokens, a_tokens))
            all_sentences.extend([q_tokens, a_tokens])
            if len(filtered_pairs)>=max_samples:
                break

    # 建词典
    vocab = Vocabulary(freq_threshold=2)
    vocab.build_vocabulary(all_sentences)
    print(f"词典大小：{len(vocab)}")

    # 划分数据集
    train_size = int(0.9*len(filtered_pairs))
    train_pairs = filtered_pairs[:train_size]
    val_pairs = filtered_pairs[train_size:]

    return train_pairs, val_pairs, vocab







def create_dataloaders(train_pairs, val_pairs, vocab, batch_size=32):
    """创建DataLoader"""
    train_dataset = DialogueDataset(train_pairs, vocab)
    val_dataset = DialogueDataset(val_pairs, vocab)

    # `lambda b: collate_fn(b, pad_idx=vocab.pad_token)` 在定义瞬间就把 `vocab.pad_token` 捕获进闭包，此后 `pad_idx` 固定不变；
    # 只有 DataLoader 每次喂进来的 `b` 会随 batch 变化。
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx=vocab.stoi[vocab.pad_token])
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx=vocab.stoi[vocab.pad_token])
    )

    return train_loader, val_loader


# ====================================
# 示例：查看数据
# ====================================
if __name__ == "__main__":
    # 加载数据（只取5000对）
    train_pairs, val_pairs, vocab = load_cornell_dialogues(
        max_samples=5000,
        max_len=15
    )

    # 查看示例
    print("\n" + "=" * 50)
    print("📝 对话示例:")
    print("=" * 50)
    for i in range(5):
        q, a = train_pairs[i]
        print(f"\nQ: {' '.join(q)}")
        print(f"A: {' '.join(a)}")

    # 创建DataLoader
    train_loader, val_loader = create_dataloaders(
        train_pairs, val_pairs, vocab, batch_size=4
    )

    # 查看一个batch
    src, tgt = next(iter(train_loader))
    print("\n" + "=" * 50)
    print("📦 Batch示例:")
    print("=" * 50)
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"\n第一个样本:")
    print(f"Source IDs: {src[0].tolist()}")
    print(f"Target IDs: {tgt[0].tolist()}")
