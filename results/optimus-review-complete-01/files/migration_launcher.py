#!/usr/bin/env python3
"""Copy verified completed detection parts into a new queue after the pinned symbol-answer fix."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hide.export_results import inspect_run
from hide.parts import (compatible_detection_sources, keyword_fix_policy, load_plan,
                        make_plan, part_path, portable_sources)
from hide.provenance import atomic_json, file_sha256, queue_lock, run_lock


def migrate(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Choose a new queue outside the original queue; existing paths are never overwritten')
    with queue_lock(source):
        old = load_plan(source/'plan.json')
        new = make_plan(old['suite'], old['part_size'])
        if not compatible_detection_sources(old['source_sha256'], new['source_sha256']):
            raise ValueError('Queue is not the exact supported pre-fix version, or other scientific code changed')
        if old['tasks'] != new['tasks']:
            raise ValueError('Task definitions changed; a keyword-fallback migration cannot change the protocol')
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.hide-upgrade-', dir=output.parent) as directory:
            staging = Path(directory)/'queue'
            staging.mkdir()
            copied, restart = [], []
            for task in old['tasks']:
                path = part_path(source, task)
                if not path.parent.parent.exists():
                    continue
                with run_lock(path):
                    report = inspect_run(path) if path.exists() else {'complete': False}
                    failed = path.with_suffix('.failed.json').exists()
                    if task['profile'] != 'timing' and report['complete'] and not failed:
                        metadata = json.loads(path.with_suffix('.manifest.json').read_text())
                        if portable_sources(metadata['source_sha256']) != old['source_sha256']:
                            raise ValueError(f'Unexpected source version in {path}')
                        partition = metadata.get('partition', {})
                        if partition.get('start') != task['start'] or partition.get('stop') != task['stop']:
                            raise ValueError(f'Unexpected partition in {path}')
                        target = part_path(staging, task)
                        shutil.copytree(path.parent.parent, target.parent.parent)
                        copied.append(dict(task=task['id'], raw_sha256=file_sha256(path),
                                           manifest_sha256=file_sha256(path.with_suffix('.manifest.json'))))
                    else:
                        # Keep old partial/error bytes for the audit; restart the part entirely under the fix.
                        backup = staging/'previous_incomplete_parts'/task['id']
                        shutil.copytree(path.parent.parent, backup)
                        for raw in backup.rglob('*.jsonl'):
                            raw.rename(raw.with_suffix('.jsonl.bak'))
                        restart.append(dict(task=task['id'], reason='timing must be remeasured' if task['profile']=='timing'
                                            else 'incomplete or failed part; original bytes retained'))
            atomic_json(staging/'plan.json', new)
            with (staging/'plan.tsv').open('w') as stream:
                stream.write('id\tprofile\tmodel\tdataset\tstart\tstop\n')
                for task in new['tasks']:
                    stream.write('\t'.join(str(task[key]) for key in ('id','profile','model','dataset','start','stop'))+'\n')
            if (source/'sessions').is_dir():
                shutil.copytree(source/'sessions', staging/'sessions')
            if (source/'overnight.log').is_file():
                shutil.copy2(source/'overnight.log', staging/'previous_overnight.log')
            record = dict(policy=keyword_fix_policy(), created_utc=datetime.now(timezone.utc).isoformat(),
                          original_queue=str(source), original_plan=old, new_plan_sha256=new['sha256'],
                          retained_parts=copied, restarted_parts=restart,
                          note='Original successful detection records/manifests/sources are byte-identical. '
                               'All incomplete and timing parts restart; original queue is retained.')
            atomic_json(staging/'migration.json', record)
            shutil.copy2(__file__, staging/'migration_launcher.py')
            shutil.move(str(staging), str(output))
    print(f'Retained {len(copied)} complete detection parts. Restart {len(restart)} previously attempted parts.')
    print(f'New plan: {output / "plan.json"}')
    print(f'Original queue preserved: {source}')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    migrate(args.source, args.output)


if __name__ == '__main__':
    main()
