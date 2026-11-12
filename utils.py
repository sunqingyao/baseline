# -*- coding: utf-8 -*-
"""通用工具：FASTA 读取、权重初始化（自包含，不依赖 Bio）。"""
import torch.nn as nn


def read_fasta_file(path):
    """读取 FASTA（或 .ss8 等同格式）文件，返回 {id: seq} 字典。

    兼容格式：
        >id
        SEQUENCE
        >id2
        SEQUENCE2
    """
    result = {}
    with open(path, 'r') as f:
        cur_id = None
        cur_seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if cur_id is not None:
                    result[cur_id] = ''.join(cur_seq)
                cur_id = line[1:].strip().split()[0]  # 取第一个空白分隔的 token 作为 id
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            result[cur_id] = ''.join(cur_seq)
    return result


def weight_init(m):
    """仅对全连接层做 Xavier 初始化；Embedding 使用其默认初始化（保证可学习）。"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
