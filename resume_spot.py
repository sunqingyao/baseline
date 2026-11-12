"""Inspect saved SPOT embeddings and run only missing feature jobs.

Run only after confirming that no other SPOT process is writing these files.
Default is read-only status; --stage prottrans/esm runs a single recovery stage.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

BASE = Path(__file__).resolve().parent
STAGES = {'prottrans': ('_pt.npy', 1024, 'generate_prottrans.py'),
          'esm': ('_esm.npy', 1280, 'generate_esm.py')}


def valid_feature(path, length, width):
    try:
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        try:
            return (array.shape == (length, width) and
                    np.issubdtype(array.dtype, np.floating) and
                    path.stat().st_size == array.offset + array.nbytes)
        finally:
            if getattr(array, '_mmap', None) is not None:
                array._mmap.close()
    except (OSError, ValueError, EOFError):
        return False


def inspect(spot, stage):
    suffix, width, _ = STAGES[stage]
    listing = spot / 'file_lists' / 'all_proteins.txt'
    entries = [line.strip() for line in listing.read_text().splitlines() if line.strip()]
    if not entries:
        raise ValueError(f'Empty file list: {listing}')
    pending, names, counts = [], set(), Counter()
    for index, entry in enumerate(entries, 1):
        if index % 5000 == 0:
            print(f'{stage}: checking {index}/{len(entries)}', flush=True)
        source = Path(entry)
        if not source.is_absolute():
            source = spot / source
        # Match the upstream generator's exact filename convention.
        name = source.name.split('.')[0]
        if name in names:
            raise ValueError(f'Output filename collision: {name}')
        names.add(name)
        lines = [line.strip() for line in source.read_text().splitlines() if line.strip()]
        if not lines or not lines[0].startswith('>') or sum(line.startswith('>') for line in lines) != 1:
            raise ValueError(f'Expected single-record FASTA: {source}')
        length = sum(len(line) for line in lines[1:])
        if not length:
            raise ValueError(f'Empty sequence: {source}')
        feature = spot / 'inputs' / (name + suffix)
        if valid_feature(feature, length, width):
            counts['complete'] += 1
        else:
            counts['invalid' if feature.exists() else 'missing'] += 1
            pending.append(entry)
    print(f'{stage}: total={len(entries)}, complete={counts["complete"]}, '
          f'missing={counts["missing"]}, invalid={counts["invalid"]}', flush=True)
    return pending


def resume(spot, stage, device):
    pending = inspect(spot, stage)
    if not pending:
        print(f'{stage}: nothing to regenerate', flush=True)
        return
    listing = spot / 'file_lists' / f'pending_{stage}.txt'
    listing.write_text(chr(10).join(pending) + chr(10))
    command = [sys.executable, '-u', STAGES[stage][2], '--file_list',
               str(listing), '--device', device]
    print(f'Running {len(pending)} pending proteins with {sys.executable}', flush=True)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    subprocess.check_call(command, cwd=spot, env=env)
    if inspect(spot, stage):
        raise RuntimeError(f'{stage}: incomplete output remains; inspect the stage log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spot-dir', type=Path, default=BASE / 'SPOT-1D-LM')
    parser.add_argument('--stage', choices=['status', *STAGES], default='status')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    spot = args.spot_dir.resolve()
    if args.stage == 'status':
        for stage in STAGES:
            inspect(spot, stage)
    else:
        resume(spot, args.stage, args.device)


if __name__ == '__main__':
    main()
