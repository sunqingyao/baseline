"""Fetch official ESM-1b weights with wget resume and validate before use.

Run when no ESM feature generation process is active.
Only the ESM checkpoint files are touched; ProtT5 caches/features are unchanged.
"""
import argparse
from datetime import datetime
import gc
from pathlib import Path
import pickle
import shutil
import subprocess

MODEL = 'esm1b_t33_650M_UR50S'
FILES = (
    (MODEL + '.pt', f'https://dl.fbaipublicfiles.com/fair-esm/models/{MODEL}.pt'),
    (MODEL + '-contact-regression.pt',
     f'https://dl.fbaipublicfiles.com/fair-esm/regression/{MODEL}-contact-regression.pt'),
)


def validate_checkpoint(path, is_main):
    import torch
    # These fixed downloads are the official ESM pickle checkpoints, including
    # argparse.Namespace in the main checkpoint; weights_only cannot parse it.
    data = torch.load(str(path), map_location='cpu', weights_only=False)
    if not isinstance(data, dict) or not isinstance(data.get('model'), dict) or not data['model']:
        raise ValueError(f'{path}: missing model state')
    if is_main:
        args = data.get('args')
        # Official legacy ESM-1b checkpoints use encoder_layers and
        # encoder_embed_dim. The ESM loader strips encoder_ before construction.
        arch = getattr(args, 'arch', None)
        layers = getattr(args, 'encoder_layers', getattr(args, 'layers', None))
        embed_dim = getattr(args, 'encoder_embed_dim', getattr(args, 'embed_dim', None))
        print(f'[MODEL] arch={arch}, layers={layers}, embed_dim={embed_dim}', flush=True)
        if arch != 'roberta_large' or layers != 33 or embed_dim != 1280:
            raise ValueError(f'{path}: unexpected ESM configuration: '
                             f'arch={arch!r}, layers={layers!r}, embed_dim={embed_dim!r}')
    else:
        expected = {'contact_head.regression.weight', 'contact_head.regression.bias'}
        if not expected.issubset(data['model']):
            raise ValueError(f'{path}: missing contact regression weights')
    del data
    gc.collect()


def backup(path):
    target = path.with_name(path.name + '.invalid.' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    path.rename(target)
    print(f'[BACKUP] {target}', flush=True)


def prepare_file(directory, name, url, wget, no_proxy=False):
    target = directory / name
    is_main = name == MODEL + '.pt'
    invalid_errors = (EOFError, RuntimeError, ValueError, pickle.UnpicklingError)
    if target.exists():
        print(f'[CHECK] {target}', flush=True)
        try:
            validate_checkpoint(target, is_main)
        except invalid_errors as exc:
            print(f'[INVALID] {exc}', flush=True)
            backup(target)
        else:
            print(f'[CACHED] {name}', flush=True)
            return target
    partial = directory / (name + '.download')
    # An interrupted validation may leave a complete .download file.
    if partial.exists():
        try:
            validate_checkpoint(partial, is_main)
        except invalid_errors:
            pass
        else:
            partial.replace(target)
            return target
    command = [wget, '--continue', '--tries=5', '--timeout=60',
               '--retry-connrefused', '--progress=dot:giga']
    if no_proxy:
        command.append('--no-proxy')
    command.extend(['--output-document', str(partial), url])
    print(f'[DOWNLOAD] {url}', flush=True)
    print(f'[PARTIAL] {partial}', flush=True)
    subprocess.check_call(command)
    print(f'[VALIDATE] Loading all tensors on CPU: {name}', flush=True)
    try:
        validate_checkpoint(partial, is_main)
    except invalid_errors as exc:
        backup(partial)
        raise RuntimeError(f'Downloaded checkpoint failed validation: {type(exc).__name__}: {exc}. '
                           'The file was backed up. Do not download again until this '
                           'specific error is diagnosed.') from exc
    partial.replace(target)
    print(f'[READY] {target}', flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-proxy', action='store_true',
                        help='Bypass proxy for these downloads only; direct access must work')
    parser.add_argument('--reuse-main', type=Path,
                        help='Validate and restore an existing main checkpoint; failure stops before any download')
    options = parser.parse_args()
    import torch
    import esm
    cache = Path(torch.hub.get_dir()) / 'checkpoints'
    cache.mkdir(parents=True, exist_ok=True)
    if options.reuse_main is not None:
        source = options.reuse_main.expanduser().resolve()
        print(f'[RECHECK] {source}; no main weight download', flush=True)
        validate_checkpoint(source, is_main=True)
        target = cache / FILES[0][0]
        if source != target.resolve():
            if target.exists():
                backup(target)
            source.replace(target)
        print(f'[RESTORED] {target}', flush=True)
    wget = shutil.which('wget')
    if not wget:
        raise SystemExit('wget was not found in the current environment.')
    for name, url in FILES:
        prepare_file(cache, name, url, wget, options.no_proxy)
    print('[CHECK] Constructing ESM-1b from local weights; no model download', flush=True)
    model_data = torch.load(cache / FILES[0][0], map_location='cpu', weights_only=False)
    regression_data = torch.load(cache / FILES[1][0], map_location='cpu', weights_only=False)
    model, alphabet = esm.pretrained.load_model_and_alphabet_core(MODEL, model_data, regression_data)
    model.eval()
    converter = alphabet.get_batch_converter()
    _, _, tokens = converter([('check', 'ACDE')])
    with torch.no_grad():
        output = model(tokens, repr_layers=[33], return_contacts=False)
    representation = output['representations'][33]
    if tuple(representation.shape) != (1, 6, 1280) or not torch.isfinite(representation).all().item():
        raise RuntimeError('ESM forward check failed')
    print('[SUCCESS] ESM weights load and CPU forward check passed.', flush=True)
    print('Next: python -u resume_spot.py --stage esm --device cuda:0', flush=True)


if __name__ == '__main__':
    main()
