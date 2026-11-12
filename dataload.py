# -*- coding: utf-8 -*-
"""数据加载器（自包含）。

输入数据：
    ss8_path       —— SPOT-1D-LM 预测的逐残基 SS8 类别，FASTA 格式（>id + 8 类字符串）。
    annotation_path—— GO 标注 CSV，第一列为蛋白 ID（作为 index），其余列为 MF/BP/CC，
                       单元格为逗号分隔的 GO 术语（空表示无标注）。
    terms_file     —— 术语表 CSV，每行两列：category,GO_term。

输出：
    每个样本返回 (ss8 类别编号张量 [max_seq_len], GO 多标签向量 [num_terms])。
    编码（One-hot / Embedding）在模型中完成，数据层只提供类别编号序列。
"""
import csv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import Config, ss8_to_idx, PAD_IDX, SS8_TO_IDX
from utils import read_fasta_file

args = Config()


class Dataload(Dataset):
    def __init__(self, ss8_path, annotation_path, terms_file, func='MF', max_seq_len=None):
        self.func = func.upper()
        self.max_seq_len = max_seq_len if max_seq_len is not None else args.max_seq_len

        self.ss8_dict = read_fasta_file(ss8_path)                 # {id: ss8 字符串}
        self._warn_unknown_ss8_chars()
        # GO 标注：index=蛋白 ID，列名含 func
        self.annotation = pd.read_csv(annotation_path, index_col=0,
                                      dtype=str, keep_default_na=False)
        if self.func not in self.annotation.columns:
            raise KeyError(f"标注文件缺少 '{self.func}' 列，现有列: {list(self.annotation.columns)}")

        self.terms_embed = self.map_annotation(terms_file, self.func)   # {GO: idx}
        self.num_terms = len(self.terms_embed)

        self.check_and_filter_data()

    # ---------------- 术语表 ----------------
    def map_annotation(self, terms_file, func):
        terms = {}
        with open(terms_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                cat = row[0].strip().upper()
                term = row[1].strip()
                if cat == func and term and term not in terms:
                    terms[term] = len(terms)
        return terms

    # ---------------- 提示：检查 SS8 字符是否与 config.SS8_CLASSES 一致 ----------------
    def _warn_unknown_ss8_chars(self):
        seen = set()
        for s in self.ss8_dict.values():
            seen.update(c.upper() for c in s)
        unknown = sorted(c for c in seen if c not in SS8_TO_IDX)
        if unknown:
            print(f'[警告] SS8 文件存在 config.SS8_CLASSES 未覆盖的字符: {unknown}，'
                  f'它们将被当作 coil(C) 处理。若这些字符代表独立类别，请在 config.py 中'
                  f'调整 SS8_CLASSES 以匹配 SPOT-1D-LM 的输出字母表。')

    # ---------------- 过滤：只保留有 SS8 且至少一个已知 GO 术语的蛋白 ----------------
    def check_and_filter_data(self):
        # 只保留存在 SS8 预测结果的蛋白
        self.annotation = self.annotation[self.annotation.index.isin(self.ss8_dict.keys())]
        # 只保留 func 列非空（至少一个术语）的行
        self.annotation = self.annotation[self.annotation[self.func].str.strip().astype(bool)]

    # ---------------- 标签 ----------------
    def annotations2y(self, annotations):
        y = torch.zeros(self.num_terms, dtype=torch.float32)
        for anno in annotations:
            anno = anno.strip()
            if anno in self.terms_embed:
                y[self.terms_embed[anno]] = 1.0
        return y

    # ---------------- 样本 ----------------
    def __getitem__(self, idx):
        protein_id = self.annotation.index[idx]
        ss8_str = self.ss8_dict[protein_id][:self.max_seq_len]

        idxs = [ss8_to_idx(c) for c in ss8_str]
        idxs = idxs + [PAD_IDX] * (self.max_seq_len - len(idxs))
        ss8 = torch.tensor(idxs, dtype=torch.long)

        label_str = str(self.annotation.iloc[idx][self.func])
        label = self.annotations2y(label_str.split(','))

        return ss8, label

    def __len__(self):
        return len(self.annotation)
