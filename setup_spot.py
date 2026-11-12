# -*- coding: utf-8 -*-
"""自动搭建 SPOT-1D-LM 环境：克隆代码 + 下载 checkpoint + 打兼容性补丁。

用法：python setup_spot.py
      python setup_spot.py --checkpoint-url <下载地址>
      python setup_spot.py --source-zip /path/to/SPOT-1D-LM-main.zip
找不到 Git 或克隆失败时自动下载源码 ZIP，也可传入本地源码 ZIP。
也可设置 SPOT_CHECKPOINT_URL 环境变量，run_all.py 会继承它。
幂等：已存在的部分会自动跳过。
"""
import argparse
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
SPOT_DIR = os.path.join(BASE, 'SPOT-1D-LM')

GIT_URL = "https://github.com/jas-preet/SPOT-1D-LM.git"
SOURCE_ZIP_URL = "https://codeload.github.com/jas-preet/SPOT-1D-LM/zip/refs/heads/main"
SOURCE_FILES = ('run_SPOT-1D-LM.sh', 'generate_esm.py', 'generate_prottrans.py',
                'run_inference.py', 'main.py')
# checkpoint 下载地址（若失效，替换成新的镜像地址）
CHECKPOINT_URL = ("http://183.36.5.251:8080/sparks_downloads/servers/downloads/"
                  "Resource/Protein/2_Protein_local_structure_prediction/spot_1d_lm.tar.gz")


def run(cmd, cwd=None):
    print(f'[运行] {cmd}')
    subprocess.check_call(cmd, shell=True, cwd=cwd)


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f'[跳过] 已存在: {dest}')
        return
    print(f'[下载] {url}')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=600) as resp, open(dest, 'wb') as f:
        total = int(resp.headers.get('Content-Length', 0) or 0)
        done = 0
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f'\r  {done * 100 // total}% ({done / 1e6:.1f}/{total / 1e6:.1f} MB)',
                      end='', flush=True)
        print()


def patch(path, old, new, name):
    with open(path, 'r', encoding='utf-8') as f:
        s = f.read()
    if old in s:
        s = s.replace(old, new)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(s)
        print(f'[补丁] {name} 已应用')
    else:
        print(f'[补丁] {name} 跳过（已应用或未找到目标）')


def source_ready(directory):
    return all((Path(directory) / name).is_file() and
               (Path(directory) / name).stat().st_size > 0 for name in SOURCE_FILES)


def extract_source_zip(archive_path, destination):
    """Extract a GitHub source archive into a staging directory."""
    destination = Path(destination)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            relative = PurePosixPath(info.filename)
            if (relative.is_absolute() or '..' in relative.parts or
                    ':' in info.filename or chr(92) in info.filename or
                    (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f'Unsafe ZIP entry: {info.filename}')
            target = destination.joinpath(*relative.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
    candidates = [destination] + [p for p in destination.iterdir() if p.is_dir()]
    roots = [p for p in candidates if source_ready(p)]
    if len(roots) != 1:
        raise ValueError('源码 ZIP 中未找到唯一且完整的 SPOT-1D-LM 目录')
    return roots[0]


def ensure_source(source_zip=None):
    target = Path(SPOT_DIR)
    if source_ready(target):
        print('[跳过] SPOT-1D-LM 源码已就绪')
        return
    if target.exists():
        raise RuntimeError(f'{target} 已存在但源码不完整，请先改名备份该目录后重试')
    with tempfile.TemporaryDirectory(prefix='.spot-source-', dir=BASE) as temporary:
        stage = Path(temporary)
        git = shutil.which('git')
        if git and not source_zip:
            clone = stage / 'clone'
            try:
                print(f'[克隆] {GIT_URL}')
                subprocess.check_call([git, 'clone', '--depth', '1', GIT_URL, str(clone)])
                if not source_ready(clone):
                    raise RuntimeError('克隆结果缺少必要源码文件')
            except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
                print(f'[克隆失败] {exc}，改用源码 ZIP')
            else:
                shutil.move(str(clone), str(target))
                return
        if source_zip:
            archive_path = Path(source_zip).resolve()
        else:
            print('[源码 ZIP] 使用 Python 下载，无需安装 Git')
            archive_path = stage / 'source.zip'
            try:
                download(SOURCE_ZIP_URL, str(archive_path))
            except OSError as exc:
                raise RuntimeError('源码 ZIP 下载失败。可在其他电脑下载源码 ZIP，上传后使用 '
                                   '--source-zip /path/to/SPOT-1D-LM-main.zip') from exc
        root = extract_source_zip(archive_path, stage / 'unpacked')
        shutil.move(str(root), str(target))
        print(f'[源码已就绪] {target}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='搭建 SPOT-1D-LM 并下载模型权重')
    parser.add_argument('--source-zip', help='本地 SPOT-1D-LM 源码 ZIP；无需 Git 或联网下载源码')
    parser.add_argument('--checkpoint-url',
                        default=os.environ.get('SPOT_CHECKPOINT_URL') or CHECKPOINT_URL,
                        help='模型权重下载地址，优先于 SPOT_CHECKPOINT_URL 环境变量')
    args = parser.parse_args(argv)
    # 1. Git 不可用或克隆失败时，自动使用官方仓库的源码 ZIP。
    ensure_source(args.source_zip)

    # 2. 下载 + 解压 checkpoint（model1.pt ~ model6.pt）
    ckpt_dir = os.path.join(SPOT_DIR, 'checkpoints')
    if all(os.path.exists(os.path.join(ckpt_dir, f'model{i}.pt')) for i in range(1, 7)):
        print('[跳过] checkpoint 已就绪')
    else:
        tar_path = os.path.join(SPOT_DIR, 'spot_1d_lm.tar.gz')
        download(args.checkpoint_url, tar_path)
        print('[解压] spot_1d_lm.tar.gz ...')
        with tarfile.open(tar_path, 'r:gz') as t:
            t.extractall(SPOT_DIR)          # 解出 spot_1d_lm/
        # 解压嵌套的 checkpoint xz（内含 checkpoints/model1.pt ~ model6.pt）
        xz_path = os.path.join(SPOT_DIR, 'spot_1d_lm', 'checkpoints', 'SPOT-LM-checkpoints.xz')
        if not os.path.exists(xz_path):
            raise FileNotFoundError(f'未找到 {xz_path}，请检查 checkpoint 压缩包结构')
        print('[解压] SPOT-LM-checkpoints.xz ...')
        with tarfile.open(xz_path, 'r:xz') as t:
            t.extractall(SPOT_DIR)          # 解出 checkpoints/model*.pt
        os.remove(tar_path) if os.path.exists(tar_path) else None

    # 3. 打兼容性补丁（适配新版 PyTorch / transformers）
    patch(os.path.join(SPOT_DIR, 'generate_esm.py'),
          'return_contacts=True', 'return_contacts=False',
          'ESM 不算 contacts（省显存）')
    patch(os.path.join(SPOT_DIR, 'generate_prottrans.py'),
          'tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True)',
          'tokenizer(sequences_Example, add_special_tokens=True, padding=True)',
          'Prottrans batch_encode_plus -> tokenizer')
    patch(os.path.join(SPOT_DIR, 'run_inference.py'),
          'num_workers=16', 'num_workers=2',
          'num_workers 降到 2')
    patch(os.path.join(SPOT_DIR, 'main.py'),
          'dict_rnam1_ASA[i] for i in single_seq',
          'dict_rnam1_ASA.get(i, 1) for i in single_seq',
          '稀有氨基酸(U/O/B/Z) KeyError')

    print('\n[完成] SPOT-1D-LM 环境已就绪')


if __name__ == '__main__':
    main()
