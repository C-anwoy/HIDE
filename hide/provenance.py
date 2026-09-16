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
    """Hash inference files, caching weight SHA-256 by path, size, mtime and ctime."""
    root = Path(path).resolve()
    if not root.is_dir():
        return {'location': str(path), 'local': False}
    cache_root = Path(os.environ.get('HIDE_CHECKPOINT_CACHE', Path.cwd()/'checkpoints'))
    cache_key = hashlib.sha256(str(root).encode()).hexdigest()
    cache_path = cache_root/'identities'/(cache_key+'.json')
    # Do not inspect unrelated original/ONNX/OpenVINO exports or Hub download bookkeeping.
    modules=[p for p in root.iterdir() if p.is_dir() and p.name.split('_')[0].isdigit()]
    files=[p for p in root.iterdir() if p.is_file() and not p.name.startswith('.')]
    files += [p for directory in modules for p in directory.rglob('*') if p.is_file() and
              not any(part.startswith('.') for part in p.relative_to(root).parts)]
    files=sorted(set(files))
    def collect(cached):
        records=[]; fresh={}; all_hashed=True
        for p in files:
            weight = p.suffix in {'.safetensors', '.bin', '.pt', '.pth'}
            if not weight and p.suffix not in {'.json', '.model', '.txt'}:
                continue
            stat=p.stat(); name=p.relative_to(root).as_posix()
            signature=[stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]
            item={'file':name,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
            old=cached.get(name,{})
            if not weight:
                item['sha256']=file_sha256(p)
            elif old.get('signature')==signature and old.get('sha256'):
                item['sha256']=old['sha256']
            elif hash_weights:
                print(f'Hashing checkpoint weight once: {p}',flush=True)
                item['sha256']=file_sha256(p)
                after=p.stat()
                if [after.st_size,after.st_mtime_ns,after.st_ctime_ns]!=signature:
                    raise ValueError(f'Checkpoint changed while hashing: {p}')
            else:
                all_hashed=False
            if weight and 'sha256' in item:
                fresh[name]={'signature':signature,'sha256':item['sha256']}
            records.append(item)
        return records,fresh,all_hashed
    def read_cached():
        return json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if hash_weights:
        with data_lock(cache_root,'identity-'+cache_key):
            records,fresh,all_hashed=collect(read_cached())
            atomic_json(cache_path,fresh)
    else:
        records,_,all_hashed=collect(read_cached())
    return {'location':str(root),'local':True,
            'weights_content_hashed':all_hashed,'files':records}


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
