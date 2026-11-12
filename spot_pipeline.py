# -*- coding: utf-8 -*-
"""SPOT-1D-LM 对接脚本：在 SPOT-1D-LM 与实验数据格式之间转换。

SPOT-1D-LM 的约定：
  - 输入 file_list：每行一个 FASTA 路径，且每个 FASTA 只含一条蛋白
  - 输出：results/<蛋白名>.csv，其中 SS8 列是逐残基 8 态二级结构（字母 C S T H G I E B）

两个子命令：
  prepare   把 data/{train,valid,test}.fasta 拆成单蛋白 FASTA + 生成合并 file_list + 映射表
  collect   把 SPOT-1D-LM 的 results/*.csv 提取 SS8 列，生成 data/{train,valid,test}.ss8

典型流程：
  python spot_pipeline.py prepare
  cd SPOT-1D-LM-main && bash run_SPOT-1D-LM.sh file_lists/all_proteins.txt cuda:0 cuda:0 cuda:0
  cd .. && python spot_pipeline.py collect
"""
import csv
import os
import re

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))


def _find_spot_dir():
    """自动识别 SPOT-1D-LM 目录名（本地解压是 SPOT-1D-LM-main，Colab git clone 是 SPOT-1D-LM）。"""
    for name in ('SPOT-1D-LM', 'SPOT-1D-LM-main'):
        d = os.path.join(BASE, name)
        if os.path.isdir(d):
            return d
    return os.path.join(BASE, 'SPOT-1D-LM')


SPOT_DIR = _find_spot_dir()
SPOT_INPUTS = os.path.join(SPOT_DIR, 'inputs')
SPOT_RESULTS = os.path.join(SPOT_DIR, 'results')
DATA_DIR = os.path.join(BASE, 'data')
MAPPING = os.path.join(BASE, 'spot_mapping.csv')
FILE_LIST = os.path.join(SPOT_DIR, 'file_lists', 'all_proteins.txt')

SPLITS = ('train', 'valid', 'test')


def read_fasta_multi(path):
    """读取多蛋白 FASTA，返回 {id: seq}。"""
    result, cur_id, cur_seq = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if cur_id is not None:
                    result[cur_id] = ''.join(cur_seq)
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            result[cur_id] = ''.join(cur_seq)
    return result


def sanitize(name):
    """把蛋白 ID 转成安全的文件名（SPOT 结果 CSV 用文件名当名字）。"""
    return re.sub(r'[^A-Za-z0-9._-]', '_', name)


def prepare():
    os.makedirs(SPOT_INPUTS, exist_ok=True)
    os.makedirs(os.path.dirname(FILE_LIST), exist_ok=True)

    used, mapping_rows, list_lines = {}, [], []
    for split in SPLITS:
        fasta = os.path.join(DATA_DIR, f'{split}.fasta')
        if not os.path.exists(fasta):
            print(f'[跳过] 缺少 {fasta}')
            continue
        for orig_id, seq in read_fasta_multi(fasta).items():
            san = sanitize(orig_id)
            if san in used and used[san] != orig_id:   # 防重名
                san = f'{san}_{len(used)}'
            used[san] = orig_id
            with open(os.path.join(SPOT_INPUTS, f'{san}.fasta'), 'w') as f:
                f.write(f'>{orig_id}\n{seq}\n')
            mapping_rows.append([san, orig_id, split])
            list_lines.append(f'inputs/{san}.fasta')

    with open(MAPPING, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sanitized', 'original', 'split'])
        w.writerows(mapping_rows)

    with open(FILE_LIST, 'w') as f:
        f.write('\n'.join(list_lines) + '\n')

    print(f'[prepare] 拆分 {len(mapping_rows)} 条蛋白 -> {SPOT_INPUTS}/')
    print(f'  file_list: {FILE_LIST}')
    print(f'  mapping:   {MAPPING}')
    print('  下一步（在 SPOT-1D-LM-main 目录内）：')
    print('    bash run_SPOT-1D-LM.sh file_lists/all_proteins.txt cuda:0 cuda:0 cuda:0')


def collect():
    if not os.path.exists(MAPPING):
        raise FileNotFoundError('缺少 spot_mapping.csv，请先运行 prepare')
    with open(MAPPING) as f:
        rows = list(csv.DictReader(f))

    out = {s: [] for s in SPLITS}
    missing = []
    for r in rows:
        san, orig, split = r['sanitized'], r['original'], r['split']
        csv_path = os.path.join(SPOT_RESULTS, f'{san}.csv')
        if not os.path.exists(csv_path):
            missing.append(san)
            continue
        df = pd.read_csv(csv_path, index_col=0)
        ss8_str = ''.join(str(v) for v in df['SS8'].values)
        out[split].append((orig, ss8_str))

    for split in SPLITS:
        path = os.path.join(DATA_DIR, f'{split}.ss8')
        with open(path, 'w') as f:
            for orig, ss8 in out[split]:
                f.write(f'>{orig}\n{ss8}\n')
        print(f'[collect] {split}: {len(out[split])} 条 -> {path}')
    if missing:
        print(f'[警告] 缺少 {len(missing)} 个结果 CSV（前10个）: {missing[:10]}')


if __name__ == '__main__':
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'prepare'
    {'prepare': prepare, 'collect': collect}[cmd]()
