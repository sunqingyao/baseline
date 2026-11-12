# -*- coding: utf-8 -*-
"""模型：SS8 编码器（One-hot / Embedding）+ 两层 MLP + GO 输出层。

对照实验设计：
    实验 A（One-hot）   —— 每个残基的 SS8 类别 -> 固定 8 维 one-hot 向量（不可学习）。
    实验 B（Embedding） —— 每个残基的 SS8 类别编号 -> 可学习的 8 维 embedding 向量。

两组实验仅编码方式不同；展平、MLP 隐藏层、GO 输出层、训练设置、评价指标完全一致。
填充位置（不足固定长度的尾部）编码为全 0 向量，两组一致。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, NUM_SS8

args = Config()


class SS8Encoder(nn.Module):
    def __init__(self, mode, num_classes=NUM_SS8, embed_dim=None):
        super().__init__()
        assert mode in ('onehot', 'embedding'), "mode 只能是 'onehot' 或 'embedding'"
        self.mode = mode
        self.num_classes = num_classes
        self.embed_dim = embed_dim if embed_dim is not None else args.embed_dim

        if mode == 'embedding':
            # 编号 0..7 为 8 类 SS8，编号 8(=num_classes) 为填充位置（零向量、不参与梯度）
            self.embedding = nn.Embedding(num_classes + 1, self.embed_dim,
                                          padding_idx=num_classes)

    def forward(self, x):
        """x: [B, L] LongTensor，取值 0..num_classes（num_classes 为填充编号）。"""
        if self.mode == 'onehot':
            # 用 num_classes+1 维 one-hot 再截断前 num_classes 维，
            # 使填充编号(num_classes)自然映射为全 0 向量。
            oh = F.one_hot(x, num_classes=self.num_classes + 1).float()
            return oh[..., :self.num_classes]          # [B, L, 8]
        else:
            return self.embedding(x)                    # [B, L, 8]，填充位为 0


class SS8MLP(nn.Module):
    """两层 MLP： 展平 -> Linear(hidden1) -> ReLU -> Linear(hidden2) -> ReLU -> GO 输出。"""

    def __init__(self, mode, num_terms, max_seq_len=None, hidden_dims=None):
        super().__init__()
        max_seq_len = max_seq_len if max_seq_len is not None else args.max_seq_len
        hidden_dims = hidden_dims if hidden_dims is not None else args.hidden_dims
        embed_dim = args.embed_dim

        self.encoder = SS8Encoder(mode, NUM_SS8, embed_dim)
        in_dim = max_seq_len * embed_dim
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.out = nn.Linear(hidden_dims[1], num_terms)

    def forward(self, x):
        e = self.encoder(x)          # [B, L, 8]
        e = self.flatten(e)          # [B, L*8]
        h = F.relu(self.fc1(e))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.out(h))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
