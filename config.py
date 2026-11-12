# -*- coding: utf-8 -*-
"""
自包含实验配置（不依赖 MMMV / esm / 外部 config 路径）。

数据流程（与实验设计一致）：
    FASTA (氨基酸序列)  --SPOT-1D-LM-->  SS8 逐残基二级结构类别
    SS8 类别 --One-hot / Embedding-->  固定长度特征矩阵 -> 展平 -> 两层 MLP -> GO 输出

说明：
    本实验的模型输入是 SS8（8 类二级结构）逐残基序列，
    FASTA 仅作为 SPOT-1D-LM 预测的原始输入，训练阶段直接使用预测得到的 SS8。
"""
import os

# DSSP 8 态二级结构类别（顺序即类别编号 0..7）
# B=β桥, C=coil(无规卷曲), E=β折叠, G=3-10螺旋, H=α螺旋, I=π螺旋, S=弯曲, T=转角
SS8_CLASSES = ['B', 'C', 'E', 'G', 'H', 'I', 'S', 'T']
SS8_TO_IDX = {c: i for i, c in enumerate(SS8_CLASSES)}
NUM_SS8 = len(SS8_CLASSES)      # 8 类
PAD_IDX = NUM_SS8               # 用于不足固定长度的填充位置编号（8）


def ss8_to_idx(ch):
    """将单个 SS8 字符映射为类别编号（0..7）。未知字符按 coil(C) 处理。"""
    return SS8_TO_IDX.get(str(ch).strip().upper(), SS8_TO_IDX['C'])


def detect_device():
    """自动检测运行设备：有 GPU 用 cuda，否则用 cpu。"""
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


class Config(object):
    # ---------------- 数据路径（相对本文件所在目录，可按需修改） ----------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')

    train_fasta = os.path.join(data_dir, 'train.fasta')
    valid_fasta = os.path.join(data_dir, 'valid.fasta')
    test_fasta = os.path.join(data_dir, 'test.fasta')

    train_ss8 = os.path.join(data_dir, 'train.ss8')
    valid_ss8 = os.path.join(data_dir, 'valid.ss8')
    test_ss8 = os.path.join(data_dir, 'test.ss8')

    annotation = os.path.join(data_dir, 'annotations.csv')   # 列: [蛋白ID(index), MF, BP, CC]
    terms_file = os.path.join(data_dir, 'terms.csv')         # 行: category,GO_term

    # ---------------- 任务与模型 ----------------
    func = 'MF'                  # GO 功能类别：'MF' / 'BP' / 'CC'
    max_seq_len = 1024           # 统一长度（截断/填充到该长度）
    embed_dim = 8                # 每个 SS8 位置的向量维度（One-hot 与 Embedding 均为 8）
    hidden_dims = (2048, 512)    # 两层 MLP 隐藏层维度（hidden1 -> hidden2 -> GO 输出层）

    # ---------------- 训练设置（两组实验共用） ----------------
    batch_size = 32
    lr = 1e-4
    weight_decay = 1e-5
    epochs = 20
    seed = 100
    device = detect_device()
    spot_device = 'cuda:0' if detect_device() == 'cuda' else 'cpu'   # SPOT-1D-LM 运行设备

    # ---------------- 输出 ----------------
    output_dir = os.path.join(base_dir, 'output')
