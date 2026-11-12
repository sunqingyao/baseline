# -*- coding: utf-8 -*-
"""一键运行完整实验：下载数据 -> 生成 SS8 -> 训练对比。

用法：
    python run_all.py                       # 全流程
    python run_all.py --skip-download        # 数据已就绪，跳过下载
    python run_all.py --skip-ss8             # SS8 已就绪，直接训练
"""
import argparse
import importlib
import os
import shlex
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
SPOT_DIR = os.path.join(BASE, 'SPOT-1D-LM')


def run(cmd, cwd=None):
    print('\n' + '=' * 72)
    print(f'[步骤] {shlex.join(cmd)}')
    print('=' * 72)
    env = os.environ.copy()
    env['PATH'] = os.path.dirname(sys.executable) + os.pathsep + env.get('PATH', '')
    subprocess.check_call(cmd, cwd=cwd, env=env)


def check_environment(needs_ss8):
    print(f'[Python] {sys.executable}', flush=True)
    try:
        torch = importlib.import_module('torch')
    except (ImportError, OSError) as exc:
        raise SystemExit(f'[环境错误] 当前 Python 无法导入 PyTorch: {exc}。'
                         '请先激活实验环境，并安装与 GPU/驱动匹配的 PyTorch。') from exc
    dependencies = {'numpy': 'numpy', 'pandas': 'pandas', 'sklearn': 'scikit-learn',
                    'matplotlib': 'matplotlib'}
    if needs_ss8:
        dependencies.update({'transformers': 'transformers', 'sentencepiece': 'sentencepiece',
                             'esm': 'fair-esm', 'tqdm': 'tqdm'})
    problems = []
    for module, package in dependencies.items():
        try:
            importlib.import_module(module)
        except (ImportError, OSError) as exc:
            problems.append(f'{package}: {exc}')
    if problems:
        raise SystemExit('[环境错误] 请在当前环境执行 python -m pip install -r requirements.txt。'
                         + chr(10) + chr(10).join(problems))
    print(f'[PyTorch] {torch.__version__}; CUDA runtime={torch.version.cuda}; '
          f'GPU available={torch.cuda.is_available()}', flush=True)
    if torch.cuda.is_available():
        print(f'[GPU] {torch.cuda.get_device_name(0)}', flush=True)
    else:
        print('[提示] PyTorch 未检测到可用 CUDA GPU，本次将使用 CPU。'
              '如需 GPU，请先检查 nvidia-smi、驱动及 PyTorch 安装。', flush=True)


def files_ready(names, sub='data'):
    return all(os.path.exists(os.path.join(BASE, sub, n)) for n in names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-download', action='store_true', help='跳过下载数据')
    parser.add_argument('--skip-ss8', action='store_true', help='跳过 SS8 生成')
    parser.add_argument('--cafa3-dir', help='本地 CAFA3 目录；保留原划分，不下载 CAFA5')
    args = parser.parse_args()
    check_environment(not args.skip_ss8 and not files_ready(['train.ss8', 'valid.ss8', 'test.ss8']))

    # ---------- 1. 下载并预处理数据 ----------
    if args.cafa3_dir:
        from prepare_cafa3 import convert
        convert(args.cafa3_dir, os.path.join(BASE, 'data'))
    elif args.skip_download:
        print('[跳过] 下载数据')
    elif files_ready(['train.fasta', 'valid.fasta', 'test.fasta',
                      'annotations.csv', 'terms.csv']):
        print('[跳过] 数据已就绪（data/ 下已有必要文件）')
    else:
        run([sys.executable, 'download_data.py'], cwd=BASE)

    # ---------- 2. 搭建 SPOT-1D-LM（克隆 + checkpoint + 补丁） ----------
    run([sys.executable, 'setup_spot.py'], cwd=BASE)

    # ---------- 3. 生成 SS8 ----------
    if args.skip_ss8:
        print('[跳过] SS8 生成')
    elif files_ready(['train.ss8', 'valid.ss8', 'test.ss8']):
        print('[跳过] SS8 已就绪')
    else:
        from config import Config
        cfg = Config()
        dev = cfg.spot_device                      # 如 'cuda:0' 或 'cpu'
        run([sys.executable, 'spot_pipeline.py', 'prepare'], cwd=BASE)     # 拆 fasta + file_list
        run(['bash', '-e', 'run_SPOT-1D-LM.sh', 'file_lists/all_proteins.txt', dev, dev, dev],
            cwd=SPOT_DIR)                                   # 生成 embedding + 预测（最耗时）
        run([sys.executable, 'spot_pipeline.py', 'collect'], cwd=BASE)     # CSV -> data/*.ss8

    # ---------- 4. 训练并对比 ----------
    run([sys.executable, 'train.py'], cwd=BASE)

    print('\n[完成] 实验全部结束，结果在 output/ 目录')


if __name__ == '__main__':
    main()
