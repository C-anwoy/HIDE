"""Repair only an incomplete trailing JSON line or trailing error; preserve the original bytes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from hide.provenance import run_lock, atomic_json, file_sha256


def recover(path, retry_error=False):
    path = Path(path).resolve()
    with run_lock(path):
        raw = path.read_bytes()
        lines = raw.splitlines(keepends=True)
        keep = len(lines)
        reason = None
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if i != len(lines)-1:
                    raise ValueError('Malformed data before the final line; investigate without automatic repair')
                keep, reason = i, 'truncated final record'
                break
            if row.get('status') != 'ok':
                if not retry_error or i != len(lines)-1:
                    raise ValueError('Error record: investigate the cause; --retry-error only permits a trailing error')
                keep, reason = i, 'explicit retry of final error record'
                break
        if reason is None:
            print('No trailing damaged/error record; nothing changed')
            return None
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        history = path.parent/'.history'/f'{path.stem}_{stamp}'
        history.mkdir(parents=True)
        backup = history/'original.jsonl.bak'
        shutil.copy2(path, backup)
        for suffix in ['.manifest.json', '.runtime.json']:
            companion = path.with_suffix(suffix)
            if companion.exists(): shutil.copy2(companion, history/companion.name)
        atomic_json(history/'recovery.json', dict(reason=reason, original_sha256=file_sha256(backup),
                    retained_lines=keep, removed_lines=len(lines)-keep))
        temporary = path.with_suffix('.recovering')
        with temporary.open('wb') as stream:
            stream.write(b''.join(lines[:keep])); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        print(f'Original retained in {history}; rerun the identical command')
        return history


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('path')
    p.add_argument('--retry-error', action='store_true')
    args=p.parse_args()
    recover(args.path, args.retry_error)


if __name__=='__main__': main()
