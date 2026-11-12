# -*- coding: utf-8 -*-
"""
自动下载 CAFA-5 数据并预处理成实验所需格式。

下载内容（Zenodo「CAFA Protein Dataset」record 10951709）：
    - train_sequences.fasta   蛋白质序列（唯一 ID + 氨基酸序列）
    - train_terms.tsv         GO 标注（长表：每行 蛋白ID -> GO术语）
    - go-basic.obo            基因本体图（用于判断每个术语属于 MF/BP/CC）

产出（data/ 目录，与 config.py 一致）：
    - annotations.csv   index=蛋白ID，列 MF/BP/CC，单元格为逗号分隔的 GO 术语
    - terms.csv         每行 category,GO_term
    - train/valid/test.fasta   划分后的序列（作为 SPOT-1D-LM 的输入）

说明：
    SS8 预测无法自动下载，需用 SPOT-1D-LM 对 train/valid/test.fasta 分别预测，
    得到 train/valid/test.ss8 后放到 data/ 即可运行 train.py（脚本末尾会再次提示）。

用法：
    python download_data.py
    python download_data.py --offline  # 仅使用本地原始文件或压缩包
"""
import argparse
import csv
import glob
import json
import os
import random
import shutil
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict

# ---------------- 可配置项 ----------------
ZENODO_BASE = "https://zenodo.org/records/10951709/files"
GO_OBO_URL = "https://current.geneontology.org/ontology/go-basic.obo"
FILES = {
    "cafa-5-protein-function-prediction.zip": f"{ZENODO_BASE}/cafa-5-protein-function-prediction.zip?download=1",
    "go-basic.obo": GO_OBO_URL,
}

RAW_DIR = "data_raw"          # 下载的原始文件目录
OUT_DIR = "data"              # 输出目录（与 config.py 的 data_dir 一致）
SPLIT = (0.7, 0.2, 0.1)       # train / valid / test 比例（对应 训练:测试:验证 = 7:1:2）
MIN_ANNOTATIONS = 5           # 术语至少被标注到多少蛋白才保留（0=全部保留）
MAX_PROTEINS = 3000           # 随机抽子集（None 或 0 = 用全部）；CPU/免费 Colab 建议几千条
SEED = 100

CATEGORIES = ('MF', 'BP', 'CC')
NS_TO_CAT = {
    'molecular_function': 'MF',
    'biological_process': 'BP',
    'cellular_component': 'CC',
}


# ---------------- 下载 ----------------
REQUIRED_FILES = ('train_sequences.fasta', 'train_terms.tsv', 'go-basic.obo')
ARCHIVE_NAME = 'cafa-5-protein-function-prediction.zip'


def valid_file(path, name=None):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    name = name or os.path.basename(path)
    if name.endswith('.zip'):
        return zipfile.is_zipfile(path)
    with open(path, 'rb') as f:
        head = f.read(8192).lstrip()
    if name.endswith('.obo'):
        return head.startswith(b'format-version:') and b'[Term]' in head
    if name.endswith('.fasta'):
        return head.startswith(b'>')
    if name.endswith('.tsv'):
        return b'\t' in head and b'GO:' in head
    return True


def download_candidates(url, name):
    yield url
    if name != ARCHIVE_NAME:
        return
    # 从记录元数据读取真实链接，不猜测镜像地址或更换数据集。
    try:
        req = urllib.request.Request(
            'https://zenodo.org/api/records/10951709',
            headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            record = json.load(resp)
        for entry in record.get('files', []):
            if entry.get('key') == name:
                for key in ('content', 'self'):
                    link = entry.get('links', {}).get(key)
                    if link and link != url:
                        yield link
    except (OSError, ValueError) as exc:
        print(f'[备用入口不可用] {exc}')


def download(url, dest):
    if valid_file(dest):
        print(f'[跳过] 已存在: {dest}')
        return
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    partial = dest + '.part'
    last_error = None
    for candidate in download_candidates(url, os.path.basename(dest)):
        print(f'[下载] {candidate}')
        try:
            req = urllib.request.Request(candidate, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=120) as resp, open(partial, 'wb') as f:
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
            if total and done != total:
                raise ValueError(f'下载不完整：应为 {total} 字节，实际 {done} 字节')
            if not valid_file(partial, os.path.basename(dest)):
                raise ValueError('文件为空、损坏或格式不符，可能下载到了错误页面')
            os.replace(partial, dest)
            print(f'\n[完成] {dest}')
            return
        except (OSError, ValueError) as exc:
            last_error = exc
            print(f'\n[下载失败] {exc}')
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
        finally:
            if os.path.exists(partial):
                os.remove(partial)
    raise RuntimeError(
        f'无法下载 {os.path.basename(dest)}：{last_error}\n'
        'HTTP 403 表示请求被拒绝，重复运行不一定有效。\n'
        f'请在能访问数据源的电脑下载 CAFA 5 原始数据，上传到 {RAW_DIR}/。\n'
        f'可放置 {ARCHIVE_NAME}，或以下三个文件（支持 Train/ 等子目录）：\n'
        f"  {', '.join(REQUIRED_FILES)}\n"
        '然后运行 python download_data.py --offline，再运行 python run_all.py。')


def collect_local_files(raw_dir):
    for name in REQUIRED_FILES:
        dest = os.path.join(raw_dir, name)
        if valid_file(dest):
            continue
        for source in sorted(glob.glob(os.path.join(raw_dir, '**', name), recursive=True)):
            if os.path.abspath(source) != os.path.abspath(dest) and valid_file(source):
                shutil.copy2(source, dest)
                print(f'[本地文件] {source} -> {dest}')
                break


def extract_required(zip_path, raw_dir):
    # 只提取需要的文件；压缩包内路径不直接作为输出路径。
    with zipfile.ZipFile(zip_path) as archive:
        for name in REQUIRED_FILES:
            dest = os.path.join(raw_dir, name)
            if valid_file(dest):
                continue
            members = [info for info in archive.infolist()
                       if not info.is_dir() and info.filename.replace('\\', '/').split('/')[-1] == name]
            if len(members) > 1:
                raise RuntimeError(f'压缩包有多个 {name}，请手动选择并放到 {raw_dir}/ 顶层')
            if not members:
                continue
            partial = dest + '.part'
            try:
                with archive.open(members[0]) as source, open(partial, 'wb') as target:
                    shutil.copyfileobj(source, target)
                if not valid_file(partial, name):
                    raise RuntimeError(f'压缩包中的 {name} 格式不正确')
                os.replace(partial, dest)
                print(f'[解压] 提取 {name}')
            finally:
                if os.path.exists(partial):
                    os.remove(partial)


# ---------------- 解析 ----------------
def read_fasta(path):
    """读 FASTA，返回 {id: seq}。"""
    result = {}
    with open(path, 'r') as f:
        cur_id, cur_seq = None, []
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


def parse_obo(path):
    """解析 go-basic.obo，返回 {GO术语: 'MF'/'BP'/'CC'}。"""
    ns_map = {}
    cur_id = None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line == '[Term]':
                cur_id = None
            elif line.startswith('id: GO:'):
                cur_id = line.split(':', 1)[1].strip()
            elif line.startswith('namespace:') and cur_id:
                ns = line.split(':', 1)[1].strip()
                cat = NS_TO_CAT.get(ns)
                if cat:
                    ns_map[cur_id] = cat
                cur_id = None
    return ns_map


def parse_terms(tsv_path, ns_map):
    """解析 train_terms.tsv（长表），返回 {蛋白ID: {'MF': set, 'BP': set, 'CC': set}}。"""
    prot_terms = defaultdict(lambda: {c: set() for c in CATEGORIES})
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, row in enumerate(reader):
            if not row:
                continue
            # 跳过表头（若存在）
            if i == 0 and row[0].strip().lower() in ('entryid', 'entry_id', 'protein', 'id'):
                continue
            if len(row) < 2:
                continue
            pid, term = row[0].strip(), row[1].strip()
            if not pid or not term:
                continue
            cat = ns_map.get(term)
            if cat:
                prot_terms[pid][cat].add(term)
    return dict(prot_terms)


# ---------------- 预处理 ----------------
def prepare(raw_dir, out_dir):
    seq_path = os.path.join(raw_dir, 'train_sequences.fasta')
    tsv_path = os.path.join(raw_dir, 'train_terms.tsv')
    obo_path = os.path.join(raw_dir, 'go-basic.obo')

    print('[解析] go-basic.obo ...')
    ns_map = parse_obo(obo_path)
    print(f'  共解析 {len(ns_map)} 个 GO 术语的 namespace')

    print('[解析] train_terms.tsv ...')
    prot_terms = parse_terms(tsv_path, ns_map)
    print(f'  共 {len(prot_terms)} 个蛋白有 GO 标注')

    print('[解析] train_sequences.fasta ...')
    sequences = read_fasta(seq_path)
    print(f'  共 {len(sequences)} 条序列')

    # 只保留既有序列又有标注的蛋白
    common = set(sequences) & set(prot_terms)
    print(f'[过滤] 序列∩标注 = {len(common)} 个蛋白')

    # 统计术语频次并过滤稀有术语
    counts = {c: Counter() for c in CATEGORIES}
    for pid in common:
        for c in CATEGORIES:
            counts[c].update(prot_terms[pid][c])
    keep = {c: {t for t, n in counts[c].items() if n >= MIN_ANNOTATIONS}
            for c in CATEGORIES}
    for c in CATEGORIES:
        print(f'  保留 {c} 术语: {len(keep[c])}（MIN_ANNOTATIONS={MIN_ANNOTATIONS}）')

    # 过滤蛋白：至少还剩下一个保留术语
    filtered = {}
    for pid in common:
        d = {c: sorted(prot_terms[pid][c] & keep[c]) for c in CATEGORIES}
        if any(d[c] for c in CATEGORIES):
            filtered[pid] = d
    print(f'[过滤] 过滤后保留 {len(filtered)} 个蛋白')

    # 划分 train/valid/test
    pids = sorted(filtered.keys())
    random.seed(SEED)
    random.shuffle(pids)
    if MAX_PROTEINS and len(pids) > MAX_PROTEINS:
        pids = pids[:MAX_PROTEINS]
        print(f'[子集] 随机抽取 {MAX_PROTEINS} 个蛋白')
    n = len(pids)
    n_train = int(n * SPLIT[0])
    n_valid = int(n * SPLIT[1])
    splits = {
        'train': pids[:n_train],
        'valid': pids[n_train:n_train + n_valid],
        'test': pids[n_train + n_valid:],
    }
    print(f'[划分] train={len(splits["train"])} valid={len(splits["valid"])} '
          f'test={len(splits["test"])}')

    os.makedirs(out_dir, exist_ok=True)

    # 术语表 terms.csv：按类别 + 排序保证索引确定
    terms_by_cat = {c: sorted(keep[c]) for c in CATEGORIES}
    with open(os.path.join(out_dir, 'terms.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        for c in CATEGORIES:
            for t in terms_by_cat[c]:
                writer.writerow([c, t])

    # 标注 annotations.csv：index=蛋白ID，列 MF/BP/CC
    with open(os.path.join(out_dir, 'annotations.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['protein', 'MF', 'BP', 'CC'])
        for pid in pids:
            writer.writerow([pid,
                             ','.join(filtered[pid]['MF']),
                             ','.join(filtered[pid]['BP']),
                             ','.join(filtered[pid]['CC'])])

    # 序列 train/valid/test.fasta（作为 SPOT-1D-LM 输入）
    for split, ids in splits.items():
        with open(os.path.join(out_dir, f'{split}.fasta'), 'w') as f:
            for pid in ids:
                f.write(f'>{pid}\n{sequences[pid]}\n')

    print(f'\n[完成] 数据已写入 {out_dir}/')
    print(f'  annotations.csv, terms.csv, train.fasta, valid.fasta, test.fasta')
    print()
    print('下一步（生成 SS8）：')
    print(f'  1) 用 SPOT-1D-LM 分别对 data/train.fasta、data/valid.fasta、data/test.fasta 预测')
    print(f'  2) 将输出的 SS8 结果按 FASTA 格式（>id + 8类字符串）保存为')
    print(f'     data/train.ss8、data/valid.ss8、data/test.ss8')
    print(f'  3) 运行 python train.py 开始实验')


def main(argv=None):
    parser = argparse.ArgumentParser(description='下载或复用本地 CAFA 5 数据并预处理')
    parser.add_argument('--offline', action='store_true', help='仅使用 data_raw/ 中的文件，不访问网络')
    args = parser.parse_args(argv)
    os.makedirs(RAW_DIR, exist_ok=True)
    collect_local_files(RAW_DIR)
    zip_path = os.path.join(RAW_DIR, ARCHIVE_NAME)
    if valid_file(zip_path) and any(
            not valid_file(os.path.join(RAW_DIR, name)) for name in REQUIRED_FILES):
        extract_required(zip_path, RAW_DIR)
    training_missing = any(not valid_file(os.path.join(RAW_DIR, name))
                           for name in REQUIRED_FILES[:2])
    if training_missing and not args.offline and not valid_file(zip_path):
        download(FILES[ARCHIVE_NAME], zip_path)
        extract_required(zip_path, RAW_DIR)
    # 优先使用数据包随附的 ontology，无需另行联网。
    obo_path = os.path.join(RAW_DIR, 'go-basic.obo')
    if not valid_file(obo_path) and not args.offline and all(
            valid_file(os.path.join(RAW_DIR, name)) for name in REQUIRED_FILES[:2]):
        download(GO_OBO_URL, obo_path)
    missing = [name for name in REQUIRED_FILES
               if not valid_file(os.path.join(RAW_DIR, name))]
    if missing:
        raise RuntimeError(f'缺少或无效的原始文件：{", ".join(missing)}。'
                           f'请放到 {RAW_DIR}/（支持子目录），然后重新运行。')

    prepare(RAW_DIR, OUT_DIR)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, zipfile.BadZipFile) as exc:
        raise SystemExit(f'[错误] {exc}') from None
