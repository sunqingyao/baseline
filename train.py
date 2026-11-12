# -*- coding: utf-8 -*-
"""训练/评测主脚本：在相同设置下对比 One-hot 与 Embedding 两种 SS8 编码方式。

用法：
    python train.py                       # 按 config.py 的默认设置运行
    python train.py --quick               # 小规模冒烟测试（验证流程可跑通）
    python train.py --func MF --epochs 20 # 指定 GO 类别 / 轮数
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from matplotlib import pyplot as plt

from config import Config
from utils import weight_init
from evaluate_tool import evaluate_result, compute_performance, compute_roc, cacul_aupr, compute_mcc, micro_score
from dataload import Dataload
from model import SS8MLP
from validate_ss8 import validate

args = Config()


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loaders():
    train_ds = Dataload(args.train_ss8, args.annotation, args.terms_file, args.func, args.max_seq_len)
    valid_ds = Dataload(args.valid_ss8, args.annotation, args.terms_file, args.func, args.max_seq_len)
    test_ds = Dataload(args.test_ss8, args.annotation, args.terms_file, args.func, args.max_seq_len)
    num_terms = train_ds.num_terms
    if num_terms == 0 or any(len(ds) == 0 for ds in (train_ds, valid_ds, test_ds)):
        raise ValueError(f'{args.func}: empty vocabulary or empty labeled train/valid/test split')

    # Sample within the existing splits, after filtering labels for this task.
    # A fresh seeded generator makes both encoding experiments use identical IDs.
    sampled = []
    for offset, (split, dataset) in enumerate(zip(('train', 'valid', 'test'),
                                                (train_ds, valid_ds, test_ds))):
        limit = getattr(args, 'max_' + split + '_samples', None)
        if limit is not None and limit < 1:
            raise ValueError(f'max_{split}_samples must be positive')
        if limit is not None and limit < len(dataset):
            generator = torch.Generator().manual_seed(args.seed + offset)
            indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
            print(f'[Subset] {split}: {len(dataset)} -> {limit}, seed={args.seed + offset}')
            dataset = Subset(dataset, indices)
        sampled.append(dataset)
    train_ds, valid_ds, test_ds = sampled

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader, num_terms, (len(train_ds), len(valid_ds), len(test_ds))


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, device):
    model.eval()
    criterion = nn.BCELoss()
    total_loss = 0.0
    all_preds, all_trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item() * x.size(0)
            all_preds.append(out.cpu().numpy())
            all_trues.append(y.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    loss = total_loss / len(loader.dataset)
    f_max, p_max, r_max, t_max, preds_bin = compute_performance(preds, trues)
    auc = compute_roc(preds, trues)
    aupr = cacul_aupr(preds, trues)
    mcc = compute_mcc(preds_bin, trues) if preds_bin is not None else 0.0
    return loss, f_max, p_max, r_max, t_max, auc, aupr, mcc



# SS71_LAYOUT_V1
def _ss71_output(base, func, tag=None):
    from pathlib import Path
    import re

    func = str(func).upper()
    if func not in {'BP', 'MF', 'CC'}:
        raise ValueError(f'未知任务：{func}')

    path = Path(base).expanduser()
    old = re.fullmatch(
        r'(BP|MF|CC)_full_\d+epochs',
        path.name,
        flags=re.IGNORECASE,
    )
    if old:
        if old.group(1).upper() != func:
            raise ValueError('输出目录的任务名称与func不一致')
        path = path.parent

    if path.name.upper() in {'BP', 'MF', 'CC'}:
        if path.name.upper() != func:
            raise ValueError('输出目录的任务名称与func不一致')
        path = path.parent / func
    else:
        path = path / func

    if tag is not None:
        mapping = {
            'a_onehot': 'onehot',
            'b_embedding': 'embed',
            'onehot': 'onehot',
            'embedding': 'embed',
            'embed': 'embed',
        }
        key = str(tag).lower()
        if key not in mapping:
            raise ValueError(f'未知实验方案：{tag}')
        path = path / mapping[key]

    return str(path)


def run_experiment(mode, save_dir):
    device = args.device
    train_loader, valid_loader, test_loader, num_terms, sizes = build_loaders()

    model = SS8MLP(mode, num_terms, max_seq_len=args.max_seq_len,
                  hidden_dims=args.hidden_dims).to(device)
    model.apply(weight_init)
    # 打印模型详细架构（每层输入输出维度 + 参数量）
    try:
        from torchinfo import summary
        summary(model, input_size=(args.batch_size, args.max_seq_len),
                dtypes=[torch.long], device=device)
    except Exception as e:
        print(model)
        print('[警告] torchinfo.summary 失败:', e)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()

    print('\n' + '=' * 80)
    print(f'实验 {"A (One-hot)" if mode == "onehot" else "B (Embedding)"} | func={args.func} '
          f'| 参数量={model.num_params():,} | train/valid/test={sizes}')
    print('=' * 80)

    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, 'results.csv')
    train_l, valid_l = [], []
    best_fmax, best_epoch = float('-inf'), 0
    best_path = os.path.join(save_dir, 'best_model.pth')

    with open(result_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'valid_loss', 'f_max', 'p_max', 'r_max',
                         't_max', 'auc', 'aupr', 'mcc'])
        for epoch in range(args.epochs):
            tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            va_loss, f_max, p_max, r_max, t_max, auc, aupr, mcc = eval_epoch(model, valid_loader, device)
            if not np.isfinite([tr_loss, va_loss, f_max]).all():
                raise ValueError('Non-finite loss or validation F-score; training stopped')

            train_l.append(tr_loss)
            valid_l.append(va_loss)

            if f_max > best_fmax:
                best_fmax, best_epoch = f_max, epoch
                torch.save(model.state_dict(), best_path)

            writer.writerow([epoch + 1, f'{tr_loss:.6f}', f'{va_loss:.6f}', f'{f_max:.6f}',
                             f'{p_max:.6f}', f'{r_max:.6f}', f'{t_max:.6f}', f'{auc:.6f}',
                             f'{aupr:.6f}', f'{mcc:.6f}'])
            print(f'epoch {epoch+1:03d}/{args.epochs} | train_loss {tr_loss:.5f} '
                  f'| valid_loss {va_loss:.5f} | F_max {f_max:.6f} | AUC {auc:.6f} '
                  f'| AUPR {aupr:.6f} | MCC {mcc:.6f}')

    # 载入最佳模型并测试
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    preds, trues = collect_predictions(model, test_loader, device)
    f_max, p_max, r_max, t_max, auc, aupr, mcc = evaluate_result(preds, trues)
    mi_p, mi_r, mi_f, _, _ = micro_score(preds, trues, threshold=0.5)

    test_result = {
        'mode': mode, 'func': args.func, 'num_params': model.num_params(),
        'f_max': f_max, 'p_max': p_max, 'r_max': r_max, 't_max': t_max,
        'auc': auc, 'aupr': aupr, 'mcc': mcc,
        'micro_p': mi_p, 'micro_r': mi_r, 'micro_f1': mi_f,
        'best_epoch': best_epoch + 1,
    }

    with open(result_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(['best_epoch', best_epoch + 1])
        writer.writerow(['test', 'f_max', 'p_max', 'r_max', 't_max', 'auc', 'aupr', 'mcc'])
        writer.writerow(['test'] + [f'{test_result[k]:.6f}' for k in
                                    ('f_max', 'p_max', 'r_max', 't_max', 'auc', 'aupr', 'mcc')])

    # 损失曲线
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_l, label='train_loss')
    plt.plot(range(1, args.epochs + 1), valid_l, label='valid_loss')
    plt.title(f'{args.func} - {mode} loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()

    return test_result


def collect_predictions(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds.append(model(x).cpu().numpy())
            trues.append(y.cpu().numpy())
    return np.concatenate(preds, 0), np.concatenate(trues, 0)


def print_comparison(results):
    headers = ['实验', 'F_max', 'P_max', 'R_max', 'AUC', 'AUPR', 'MCC',
               'micro_P', 'micro_R', 'micro_F1', '参数量']
    rows = []
    for mode, tag in [('onehot', 'A One-hot'), ('embedding', 'B Embedding')]:
        r = results[mode]
        rows.append([tag, r['f_max'], r['p_max'], r['r_max'], r['auc'], r['aupr'],
                     r['mcc'], r['micro_p'], r['micro_r'], r['micro_f1'], r['num_params']])

    print('\n' + '=' * 100)
    print(f'实验结果对比（func={args.func}）')
    print('=' * 100)
    header_line = ' | '.join(f'{h:>12}' for h in headers)
    print(header_line)
    print('-' * len(header_line))
    for row in rows:
        line = ' | '.join(f'{v:>12}' if isinstance(v, str) else f'{v:>12.6f}' for v in row[:-1])
        line += f' | {row[-1]:>12,}'
        print(line)
    print('=' * 100)

    # 保存对比表
    cmp_path = os.path.join(_ss71_output(args.output_dir, args.func), 'comparison.csv')
    os.makedirs(_ss71_output(args.output_dir, args.func), exist_ok=True)
    with open(cmp_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)
    print(f'对比结果已保存到: {cmp_path}')


def main():
    if args.epochs < 1:
        raise ValueError('epochs must be positive')
    validate(args.data_dir)
    set_seed(args.seed)
    results = {}
    for mode in ('onehot', 'embedding'):
        tag = 'A_onehot' if mode == 'onehot' else 'B_embedding'
        save_dir = _ss71_output(args.output_dir, args.func, tag)
        results[mode] = run_experiment(mode, save_dir)
    print_comparison(results)
    print('\nDone!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, default=None, help='GO 类别: MF/BP/CC')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--quick', action='store_true', help='小规模冒烟测试')
    parser.add_argument('--max-train-samples', type=int, help='训练集有效样本上限；固定种子抽样')
    parser.add_argument('--max-valid-samples', type=int, help='验证集有效样本上限；默认使用全量')
    parser.add_argument('--max-test-samples', type=int, help='测试集有效样本上限；默认使用全量')
    parser.add_argument('--output-dir', help='单独的实验输出目录，避免覆盖正式结果')
    opt = parser.parse_args()

    for split in ('train', 'valid', 'test'):
        limit = getattr(opt, 'max_' + split + '_samples')
        if limit is not None and limit < 1:
            parser.error(f'--max-{split}-samples must be positive')
        setattr(args, 'max_' + split + '_samples', limit)
    if opt.output_dir:
        args.output_dir = os.path.abspath(opt.output_dir)
    elif any(getattr(opt, 'max_' + split + '_samples') is not None
             for split in ('train', 'valid', 'test')):
        tag = '_'.join(split + str(getattr(opt, 'max_' + split + '_samples') or 'all')
                       for split in ('train', 'valid', 'test'))
        args.output_dir = os.path.join(args.output_dir, 'subset_' + tag)

    if opt.func:
        args.func = opt.func.upper()
    if opt.epochs is not None:
        args.epochs = opt.epochs
    if opt.quick:
        args.output_dir = os.path.join(args.output_dir, 'quick_check')
        args.max_seq_len = 64
        args.hidden_dims = (64, 32)
        args.batch_size = 32
        args.epochs = 3

    main()
