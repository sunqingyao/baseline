"""Validate SS8 coverage and per-residue alignment before training."""
from pathlib import Path


def read_records(path):
    records, pid = {}, None
    with Path(path).open(encoding='utf-8-sig') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                fields = line[1:].split()
                if not fields or fields[0] in records:
                    raise ValueError(f'{path}: empty or duplicate ID')
                pid = fields[0]
                records[pid] = ''
            elif pid is None:
                raise ValueError(f'{path}: sequence without header')
            else:
                records[pid] += line
    if not records or any(not seq for seq in records.values()):
        raise ValueError(f'{path}: empty records')
    return records


def validate(data_dir):
    data_dir = Path(data_dir)
    seen, counts = set(), {}
    for split in ('train', 'valid', 'test'):
        sequences = read_records(data_dir / (split + '.fasta'))
        structures = read_records(data_dir / (split + '.ss8'))
        missing = set(sequences) - set(structures)
        extra = set(structures) - set(sequences)
        if missing or extra:
            raise ValueError(f'{split}: missing SS8={len(missing)}, unexpected SS8={len(extra)}; '
                             f'missing examples={sorted(missing)[:5]}')
        if seen & set(sequences):
            raise ValueError(f'{split}: protein IDs overlap another split')
        seen.update(sequences)
        for pid, seq in sequences.items():
            ss8 = structures[pid]
            if len(ss8) != len(seq):
                raise ValueError(f'{split}/{pid}: sequence length {len(seq)} != SS8 length {len(ss8)}')
            if set(ss8) - set('BCEGHIST'):
                raise ValueError(f'{split}/{pid}: invalid SS8 symbols')
        counts[split] = len(sequences)
    print(f'[SS8 validation passed] {counts}')
    return counts


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'data')
    options = parser.parse_args()
    try:
        validate(options.data_dir)
    except (OSError, ValueError) as exc:
        raise SystemExit(f'SS8 validation failed: {exc}') from None
