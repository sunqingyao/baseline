# -*- coding: utf-8 -*-
"""一键运行完整实验：下载数据 -> 生成 SS8 -> 训练对比。

用法：
    python run_all.py                       # 全流程
    python run_all.py --skip-download        # 数据已就绪，跳过下载
    python run_all.py --skip-ss8             # SS8 已就绪，直接训练
"""
import argparse
import os
import subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
SPOT_DIR = os.path.join(BASE, 'SPOT-1D-LM')


def run(cmd, cwd=None):
    print('\n' + '=' * 72)
    print(f'[步骤] {cmd}')
    print('=' * 72)
    subprocess.check_call(cmd, shell=True, cwd=cwd)


def files_ready(names, sub='data'):
    return all(os.path.exists(os.path.join(BASE, sub, n)) for n in names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-download', action='store_true', help='跳过下载数据')
    parser.add_argument('--skip-ss8', action='store_true', help='跳过 SS8 生成')
    args = parser.parse_args()

    # ---------- 1. 下载并预处理数据 ----------
    if args.skip_download:
        print('[跳过] 下载数据')
    elif files_ready(['train.fasta', 'valid.fasta', 'test.fasta',
                      'annotations.csv', 'terms.csv']):
        print('[跳过] 数据已就绪（data/ 下已有必要文件）')
    else:
        run('python download_data.py', cwd=BASE)

    # ---------- 2. 搭建 SPOT-1D-LM（克隆 + checkpoint + 补丁） ----------
    run('python setup_spot.py', cwd=BASE)

    # ---------- 3. 生成 SS8 ----------
    if args.skip_ss8:
        print('[跳过] SS8 生成')
    elif files_ready(['train.ss8', 'valid.ss8', 'test.ss8']):
        print('[跳过] SS8 已就绪')
    else:
        from config import Config
        cfg = Config()
        dev = cfg.spot_device                      # 如 'cuda:0' 或 'cpu'
        run('python spot_pipeline.py prepare', cwd=BASE)     # 拆 fasta + file_list
        run(f'bash run_SPOT-1D-LM.sh file_lists/all_proteins.txt {dev} {dev} {dev}',
            cwd=SPOT_DIR)                                   # 生成 embedding + 预测（最耗时）
        run('python spot_pipeline.py collect', cwd=BASE)     # CSV -> data/*.ss8

    # ---------- 4. 训练并对比 ----------
    run('python train.py', cwd=BASE)

    print('\n[完成] 实验全部结束，结果在 output/ 目录')


if __name__ == '__main__':
    main()
