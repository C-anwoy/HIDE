#!/usr/bin/env python3
"""Resume a fixed 2,000-example paired comparison using existing answer-run parts."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hide import answer_runs, parts
from hide.export_results import inspect_run, verify
from hide.provenance import atomic_json, queue_lock, run_lock
from scripts.answer_worker import wait_for_initialization

LIMIT = 2000


def selected_tasks(plan):
    tasks = [t for t in plan['tasks'] if t['profile'] == 'answer-qa' and t['stop'] <= LIMIT]
    for model in answer_runs.MODELS:
        for dataset in ['nq_open', 'triviaqa', 'SQuAD', 'race']:
            cohort = sorted((t for t in tasks if t['model'] == model and t['dataset'] == dataset),
                            key=lambda t: t['start'])
            cursor = 0
            for task in cohort:
                if task['start'] != cursor:
                    raise ValueError('Subset does not have contiguous existing parts')
                cursor = task['stop']
            if cursor != LIMIT:
                raise ValueError('Existing plan cannot provide exactly 2,000 examples per cell')
    # Complete factuality first, alternating its datasets at each part boundary.
    order = {'nq_open': 0, 'triviaqa': 1, 'SQuAD': 2, 'race': 3}
    return sorted(tasks, key=lambda t: (order[t['dataset']]//2, t['start'],
                                        order[t['dataset']], t['model']))


def selection(root):
    plan = parts.load_plan(root/'plan.json', check_source=True)
    if plan['suite'] != 'answer-detection':
        raise ValueError('Use the existing corrected answer-detection queue')
    tasks = selected_tasks(plan)
    spec = dict(format='hide-fixed-subset-v1', parent_plan_sha256=plan['sha256'],
                examples_per_dataset_per_model=LIMIT, seed=42,
                selection='Positions [0, 2000) in the existing seed-42 shuffled cohort; '
                          'chosen for the deadline after earlier full-cohort results were inspected; '
                          'no selection using individual detector scores or correctness labels.',
                answer_boundary='first-line', tasks=tasks)
    with queue_lock(root, shared=True), run_lock(root/'subsets/answer-2000.json'):
        path = root/'subsets/answer-2000.json'
        if path.exists() and json.loads(path.read_text()) != spec:
            raise ValueError('Saved subset specification differs; do not overwrite it')
        atomic_json(path, spec)
        source = Path(__file__).read_bytes()
        snapshot = root/'subsets'/('launcher-'+hashlib.sha256(source).hexdigest()+'.py')
        snapshot.write_bytes(source)
    return tasks


def status(root, tasks):
    counts = dict(total=len(tasks), complete=0, partial=0, missing=0, failed=0, running=0)
    for task in tasks:
        path = parts.part_path(root, task)
        try:
            with ExitStack() as stack:
                stack.enter_context(run_lock(root/'claims'/f'{task["id"]}.claim'))
                stack.enter_context(run_lock(path))
                if path.with_suffix('.failed.json').exists():
                    counts['failed'] += 1
                elif not path.exists():
                    counts['missing'] += 1
                elif inspect_run(path)['complete']:
                    counts['complete'] += 1
                else:
                    counts['partial'] += 1
        except RuntimeError as exc:
            if not isinstance(exc.__cause__, BlockingIOError):
                raise
            counts['running'] += 1
    print(json.dumps(counts, indent=2), flush=True)
    return counts


def run(root, model, tasks, hours, hard_hours):
    original_choose = parts.choose_tasks
    original_init = answer_runs.initialize
    old_argv = sys.argv
    ids = {t['id'] for t in tasks}
    ranks = {t['id']: i for i, t in enumerate(tasks)}

    def choose(plan, kind='detection', index=0, workers=1):
        chosen = original_choose(plan, kind, index, workers)
        if plan['suite'] == 'answer-pilots':
            return chosen
        if plan['suite'] != 'answer-detection' or kind != 'detection':
            raise ValueError('Subset worker only supports corrected detection')
        return sorted((t for t in chosen if t['id'] in ids), key=lambda t: ranks[t['id']])

    parts.choose_tasks = choose
    answer_runs.initialize = wait_for_initialization(original_init)
    sys.argv = ['answer-subset', 'detection', '--root', str(root), '--models', model,
                '--hours', str(hours), '--hard-hours', str(hard_hours)]
    try:
        answer_runs.main()
    finally:
        parts.choose_tasks = original_choose
        answer_runs.initialize = original_init
        sys.argv = old_argv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['status', 'run', 'export'])
    p.add_argument('--root', type=Path, default=Path('outputs/answer-detection'))
    p.add_argument('--model', choices=answer_runs.MODELS)
    p.add_argument('--gpu', default='1', help='Physical GPU index or UUID; run only')
    p.add_argument('--hours', type=float, default=8)
    p.add_argument('--hard-hours', type=float, default=9)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.action == 'run' and (not args.model or not 0 < args.hours < args.hard_hours <= 22):
        p.error('Run requires --model and 0 < hours < hard-hours <= 22')
    if args.action == 'export' and not args.output:
        p.error('Export requires --output')
    root = args.root.resolve()
    tasks = selection(root)
    if args.action == 'status':
        status(root, tasks)
    elif args.action == 'run':
        gpu = subprocess.check_output(['nvidia-smi', '-i', args.gpu, '--query-gpu=uuid',
                                       '--format=csv,noheader'], text=True).strip()
        if not gpu.startswith('GPU-') or '\n' in gpu:
            raise ValueError('Expected exactly one GPU UUID')
        os.environ.update(CUDA_VISIBLE_DEVICES=gpu, HIDE_DEVICE='cuda:0')
        with run_lock(root/'subsets'/f'{args.model}.claim'):
            run(root, args.model, tasks, args.hours, args.hard_hours)
        status(root, [t for t in tasks if t['model'] == args.model])
    else:
        # Hold the full queue exclusively so the checked subset cannot change before export.
        with queue_lock(root):
            counts = status(root, tasks)
            if counts['complete'] != counts['total']:
                raise ValueError('The 2,000-example comparison is not complete; retain and resume it')
            answer_runs.check_pilots(root, answer_runs.MODELS)
            # The enclosing original full-cohort plan intentionally remains incomplete.
            from hide.export_results import _export
            with ExitStack() as stack:
                for path in sorted(root.rglob('*.jsonl')):
                    stack.enter_context(run_lock(path))
                _export(root, args.output, allow_incomplete=True)
        verify(args.output)
        print('Verified export: fixed subset complete; original full-cohort plan may be incomplete.')


if __name__ == '__main__':
    main()
