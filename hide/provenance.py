"""Run identity, atomic metadata writes and single-writer protection."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False, default=str)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def run_lock(output):
    """POSIX advisory lock; automatically released on exit, including process death."""
    import fcntl
    path = Path(output).with_suffix('.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another process is writing {output}') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def example_seed(seed, dataset, example_id, repeat=0):
    raw = json.dumps([seed, dataset, str(example_id), repeat]).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], 'big')


def source_files():
    root = Path(__file__).resolve().parent.parent
    files = list((root / 'hide').rglob('*.py'))
    files += list((root / 'hide' / 'config').glob('*.json'))
    files += [root / 'pyproject.toml']
    return {str(p): file_sha256(p) for p in sorted(files) if p.is_file()}


def checkpoint_identity(path, hash_weights=False):
    """Hash tokenizer/config files; inventory weights, optionally hash their bytes."""
    root = Path(path)
    if not root.is_dir():
        return {'location': str(path), 'local': False}
    records = []
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        weight = p.suffix in {'.safetensors', '.bin', '.pt', '.pth'}
        if not weight and p.suffix not in {'.json', '.model', '.txt'}:
            continue
        stat = p.stat()
        item = {'file': p.name, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        if not weight or hash_weights:
            item['sha256'] = file_sha256(p)
        records.append(item)
    return {'location': str(root.resolve()), 'local': True,
            'weights_content_hashed': hash_weights, 'files': records}


@contextmanager
def queue_lock(root, shared=False):
    """Workers share this lock; merging/exporting requires a stopped queue."""
    import fcntl
    path = Path(root)/'.queue.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Queue is active; stop workers before merging/exporting, or finish the snapshot first') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def data_lock(root, dataset):
    """Serialize cache initialization/mapping so concurrent workers cannot corrupt it."""
    import fcntl
    path = Path(root)/'.locks'/f'{dataset}.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)
