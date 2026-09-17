#!/usr/bin/env bash
# Schedule existing, immutable review-plan tasks; scientific protocols are unchanged.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
exec python -u - "$@" <<'PY'
import argparse
import json
import math
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time
import uuid

from hide.export_results import inspect_run
from hide.parts import load_plan, part_path
from hide.provenance import run_lock

parser = argparse.ArgumentParser(description='One-GPU reviewer-priority queue; resume with the same command.')
parser.add_argument('--plan', default='outputs/review/plan.json')
parser.add_argument('--queue', default='outputs/review')
parser.add_argument('--hours', type=float, default=18)
parser.add_argument('--hard-hours', type=float, default=20)
parser.add_argument('--kind', choices=['all', 'detection', 'timing'], default='all',
                    help='Use detection while timing cannot have exclusive GPU access.')
parser.add_argument('--max-parts', type=int, default=0, help='Stop after this many new completed parts; 0 means unlimited.')
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()
if not (math.isfinite(args.hours) and math.isfinite(args.hard_hours)
        and 0 < args.hours < args.hard_hours <= 22) or args.max_parts < 0:
    parser.error('Require 0 < hours < hard-hours <= 22 and max-parts >= 0')
plan = load_plan(args.plan, check_source=True)
if plan['suite'] != 'review':
    parser.error('This launcher requires a review plan. Do not duplicate an existing full-plan run.')

def priority(task):
    # Alternate models after each part within a dataset, preserving both cohorts.
    stage = 2 if task['profile'] == 'timing' else {'nq_open': 0, 'SQuAD': 1, 'triviaqa': 3, 'race': 4}[task['dataset']]
    if stage == 2:
        model = ['llama3-3b', 'llama3-8b', 'gemma-2-9b', 'gemma-2-27b'].index(task['model'])
        return stage, model, task['dataset']
    return stage, task['start'], ['llama3-8b', 'gemma-2-9b'].index(task['model'])

tasks = sorted(plan['tasks'], key=priority)
if args.kind != 'all':
    tasks = [task for task in tasks if (task['profile'] == 'timing') == (args.kind == 'timing')]
if args.dry_run:
    for task in tasks:
        print(task['id'])
    sys.exit(0)

queue = Path(args.queue).resolve()
queue.mkdir(parents=True, exist_ok=True)
started = time.monotonic()
soft, hard = started + args.hours * 3600, started + args.hard_hours * 3600
requested = False
child = None
guard_directory = None

def save_guard_failure(task, attempt, error):
    global guard_directory
    if guard_directory is None:
        guard_directory = queue / 'sessions' / ('timing_guard_' + uuid.uuid4().hex)
        guard_directory.mkdir(parents=True)
        (guard_directory / 'launcher.sh').write_bytes(Path('scripts/run_priority.sh').read_bytes())
        (guard_directory / 'timing_worker.py').write_bytes(Path('scripts/timing_worker.py').read_bytes())
    report = {'task': task['id'], 'attempt': attempt, 'unix_time': time.time(),
              'worker_stderr': error, 'plan_sha256': plan.get('sha256')}
    for line in error.splitlines():
        if line.startswith('TIMING_GUARD_DIAGNOSTICS='):
            report['failed_check'] = json.loads(line.split('=', 1)[1])
    try:
        import os
        expected = json.loads((queue / 'timing_device.json').read_text())
        selected = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version,power.limit',
                                        '--format=csv,noheader,nounits'], text=True, timeout=5)
        matches = [line.strip() for line in rows.splitlines() if line.split(',')[0].strip() == selected]
        current = {'hostname': platform.node(), 'device': matches[0] if len(matches) == 1 else matches,
                   'plan_sha256': plan.get('sha256')}
        report.update(expected=expected, observed_after_failure=current,
                      differences={key: {'expected': expected.get(key), 'observed': current.get(key)}
                                   for key in set(expected) | set(current) if expected.get(key) != current.get(key)},
                      note='This snapshot follows the failed check; a transient discrepancy may already have cleared.')
    except Exception as exc:
        report['diagnostic_error'] = f'{type(exc).__name__}: {exc}'
    path = guard_directory / f'{task["id"]}_attempt{attempt}.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Timing preflight attempt {attempt}/6 failed. Exact diagnostics: {path}', flush=True)

def stop(signum, frame):
    global requested
    requested = True
    if child is not None and child.poll() is None:
        try:
            child.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
completed = 0
with run_lock(queue / 'priority-launcher.claim'):
    for task in tasks:
        now = time.monotonic()
        if requested or now >= soft or (args.max_parts and completed >= args.max_parts):
            break
        output = part_path(queue, task)
        if output.with_suffix('.failed.json').exists():
            sys.exit(f'Failed part needs inspection/retry before continuing: {task["id"]}')
        if output.exists() and inspect_run(output)['complete']:
            continue
        for attempt in range(1, 7):
            now = time.monotonic()
            if requested or now >= soft:
                break
            entry = ([sys.executable, '-u', 'scripts/timing_worker.py'] if task['profile'] == 'timing'
                     else [sys.executable, '-u', '-m', 'hide.parts', 'work'])
            cmd = entry + ['--plan', args.plan,
                   '--queue', str(queue), '--task', task['id'],
                   '--kind', 'timing' if task['profile'] == 'timing' else 'detection',
                   '--hours', str((soft-now)/3600), '--hard-hours', str((hard-now)/3600)]
            print(f'PRIORITY {task["id"]}; {(soft-now)/3600:.2f} work hours remaining', flush=True)
            with tempfile.TemporaryFile(mode='w+') as errors:
                child = subprocess.Popen(cmd, stderr=errors)
                code = child.wait()
                child = None
                errors.seek(0)
                error = errors.read()
            guard_failure = (code != 0 and task['profile'] == 'timing'
                             and 'ValueError: Timing worker GPU, host, driver or power limit changed' in error
                             and not output.with_suffix('.failed.json').exists())
            if guard_failure:
                save_guard_failure(task, attempt, error)
                if attempt < 6 and not requested and time.monotonic() < soft:
                    print('Waiting five seconds, then checking again in a fresh worker.', flush=True)
                    time.sleep(min(5, max(0, soft-time.monotonic())))
                    continue
            if error:
                print(error, file=sys.stderr, end='', flush=True)
            if code:
                sys.exit(f'Worker failed ({code}); inspect the printed task log. Saved results are retained.')
            break
        if not output.exists() or not inspect_run(output)['complete']:
            print('Part paused or claimed elsewhere; stopping. Resume on this GPU with the same command.')
            break
        completed += 1
        if task['profile'] == 'timing' and not requested:
            # Teardown delay is outside measurements, but counts toward the shared worker budget.
            time.sleep(min(5, max(0, soft-time.monotonic())))
print(f'Priority launcher stopped: {completed} new parts complete. Use parts.sh status for full coverage.')
PY
