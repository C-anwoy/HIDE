#!/usr/bin/env bash
# Schedule existing, immutable review-plan tasks; scientific protocols are unchanged.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
exec python -u - "$@" <<'PY'
import argparse
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

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
        cmd = [sys.executable, '-u', '-m', 'hide.parts', 'work', '--plan', args.plan,
               '--queue', str(queue), '--task', task['id'],
               '--kind', 'timing' if task['profile'] == 'timing' else 'detection',
               '--hours', str((soft-now)/3600), '--hard-hours', str((hard-now)/3600)]
        print(f'PRIORITY {task["id"]}; {(soft-now)/3600:.2f} work hours remaining', flush=True)
        child = subprocess.Popen(cmd)
        code = child.wait()
        child = None
        if code:
            sys.exit(f'Worker failed ({code}); inspect the printed task log. Saved results are retained.')
        if not output.exists() or not inspect_run(output)['complete']:
            print('Part paused or claimed elsewhere; stopping. Resume on this GPU with the same command.')
            break
        completed += 1
print(f'Priority launcher stopped: {completed} new parts complete. Use parts.sh status for full coverage.')
PY
