"""Pilot-gated, resumable first-answer-line experiments, with a separate result queue."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from hide.answer_boundary import first_answer_line
from hide.export_results import inspect_run
from hide.parts import is_timing, load_plan, make_plan, part_path, work
from hide.provenance import atomic_json, file_sha256, queue_lock, run_lock

MODELS = ['llama3-8b', 'gemma-2-9b']


def initialize(root):
    root = Path(root).resolve()
    for path, suite in [(root/'plan.json', 'answer-review'),
                        (root/'pilot/plan.json', 'answer-pilots')]:
        with run_lock(path):
            if path.exists():
                if load_plan(path, check_source=True)['suite'] != suite:
                    raise ValueError(f'Wrong suite at {path}; choose a new output root')
            else:
                atomic_json(path, make_plan(suite))
    return root


def check_pilots(root, models):
    queue = Path(root)/'pilot'
    plan = load_plan(queue/'plan.json', check_source=True)
    summaries = []
    with queue_lock(queue, shared=True):
        for task in plan['tasks']:
            if task['model'] not in models:
                continue
            path = part_path(queue, task)
            if not path.exists():
                raise ValueError(f'Pilot missing: {task["id"]}')
            with run_lock(path):
                report = inspect_run(path)
                meta = json.loads(path.with_suffix('.manifest.json').read_text())
                if not report['complete'] or meta['arguments'].get('answer_boundary') != 'first-line':
                    raise ValueError(f'Pilot incomplete or wrong protocol: {path}')
                from hide.parts import portable_sources
                if portable_sources(meta['source_sha256']) != plan['source_sha256']:
                    raise ValueError(f'Pilot source differs from the new plan: {path}')
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                issues = []
                for row in rows:
                    answer, reached, suffix = first_answer_line(row['generated_text'])
                    if (row.get('answer_boundary') != 'first-line' or row.get('evaluated_text') != answer
                            or row.get('answer_boundary_reached') != reached
                            or row.get('boundary_token_suffix') != suffix):
                        issues.append(f'{row["id"]}: boundary metadata mismatch')
                    if '\n' in answer or '\r' in answer:
                        issues.append(f'{row["id"]}: multiline evaluated answer')
                    if suffix.strip():
                        issues.append(f'{row["id"]}: text after boundary inside terminal token; inspect tokenizer')
                    for key in ['HIDE_score', 'Omega', 'Delta_in', 'sentence_similarity', 'rouge_l']:
                        value = row.get(key)
                        if not isinstance(value, (int, float)) or not math.isfinite(value):
                            issues.append(f'{row["id"]}: nonfinite/missing {key}')
                summary = dict(task=task['id'], raw_sha256=file_sha256(path), n=len(rows),
                               passed=not issues, issues=issues,
                               line_stops=sum(r.get('answer_boundary_reached', False) for r in rows),
                               no_output_state=sum(r.get('no_output_state', False) for r in rows),
                               empty_answers=sum(not r.get('evaluated_text') for r in rows),
                               generation_cap=sum(r.get('hit_generation_cap', False) for r in rows),
                               n_correct=sum(r.get('is_correct', 0) for r in rows),
                               samples=[{k: r.get(k) for k in ['id','question','answer','generated_text',
                                           'evaluated_text','sentence_similarity']} for r in rows])
                atomic_json(queue/'checks'/f'{task["model"]}_{task["dataset"]}.json', summary)
                summaries.append(summary)
                print(f'{task["model"]}/{task["dataset"]}: {len(rows)} pilot rows; '
                      f'empty={summary["empty_answers"]}, cap={summary["generation_cap"]}; '
                      f'check={summary["passed"]}', flush=True)
    failures = [s for s in summaries if not s['passed']]
    if failures:
        raise ValueError('Pilot gate stopped; all pilot records retained. Inspect pilot/checks/: '
                         + '; '.join(s['task']+': '+', '.join(s['issues'][:3]) for s in failures))
    print(f'PASS: {len(summaries)} pilot datasets; no accuracy/AUC threshold was applied.', flush=True)
    return summaries


class TimingMonitor:
    """Audit samples outside measured calls; not a guarantee of continuous exclusivity."""
    def __init__(self, root):
        self.path = Path(root)/'telemetry'/f'timing-{uuid.uuid4().hex}.ndjson'
        self.stop = threading.Event()

    def sample(self):
        row = dict(utc=datetime.now(timezone.utc).isoformat(), host_load_1_5_15=os.getloadavg(),
                   cpu_count=os.cpu_count(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        try:
            row['gpu'] = subprocess.check_output([
                'nvidia-smi', '-i', os.environ['CUDA_VISIBLE_DEVICES'],
                '--query-gpu=uuid,name,driver_version,power.limit,power.draw,temperature.gpu,clocks.current.sm,utilization.gpu,memory.used',
                '--format=csv,noheader,nounits'], text=True, timeout=5).strip()
            row['gpu_columns'] = ['uuid','name','driver_version','power_limit_w','power_draw_w',
                                  'temperature_c','sm_clock_mhz','utilization_pct','memory_used_mib']
        except Exception as exc:
            row['monitor_error'] = f'{type(exc).__name__}: {exc}'
        with self.path.open('a') as stream:
            stream.write(json.dumps(row)+'\n')

    def loop(self):
        while not self.stop.wait(30):
            self.sample()

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join(timeout=6)
        self.sample()


def bounded_work(root, models, pilot, start, hours, hard_hours, workers=1, worker_index=0, datasets=None):
    elapsed = (time.monotonic()-start)/3600
    if elapsed >= hours:
        print('Budget reached; rerun the same command to resume.', flush=True)
        return False
    queue = Path(root)/'pilot' if pilot else Path(root)
    work(argparse.Namespace(plan=str(queue/'plan.json'), queue=str(queue), kind='detection',
         models=models, profiles=['answer-pilot' if pilot else 'answer-qa'], task=None,
         max_parts=0, workers=1 if pilot else workers, worker_index=0 if pilot else worker_index,
         datasets=None if pilot else datasets,
         hours=hours-elapsed, hard_hours=hard_hours-elapsed))
    return time.monotonic()-start < hours*3600


def wait_detection(root, deadline):
    """Wait for complete, unlocked parts; cache immutable-file inspections by stat."""
    plan = load_plan(Path(root)/'plan.json', check_source=True)
    tasks = [t for t in plan['tasks'] if not is_timing(t['profile'])]
    cache = {}
    previous = None
    while time.monotonic() < deadline:
        failed_pilots = list((Path(root)/'pilot/runs').glob('*/*/*.failed.json'))
        if failed_pilots:
            raise RuntimeError(f'Pilot failed; timing waiter stops: {failed_pilots[0]}')
        for check in (Path(root)/'pilot/checks').glob('*.json'):
            if not json.loads(check.read_text()).get('passed', False):
                raise RuntimeError(f'Pilot gate failed; timing waiter stops: {check}')
        complete = 0
        for task in tasks:
            path = part_path(root, task)
            if path.with_suffix('.failed.json').exists():
                raise RuntimeError(f'Detection failed: {task["id"]}; timing waiter stops for inspection')
            if not path.exists():
                continue
            try:
                with run_lock(Path(root)/'claims'/f'{task["id"]}.claim'), run_lock(path):
                    stamp = (path.stat().st_size, path.stat().st_mtime_ns)
                    if cache.get(task['id']) != stamp:
                        if not inspect_run(path)['complete']:
                            continue
                        cache[task['id']] = stamp
                    complete += 1
            except RuntimeError:
                continue  # A live detection writer still owns this part.
        if complete != previous:
            print(f'Waiting for detection: {complete}/{len(tasks)} complete and unlocked.', flush=True)
            previous = complete
        if complete == len(tasks):
            return True
        time.sleep(min(15, max(0, deadline-time.monotonic())))
    print('Waiting budget reached; rerun the timing command to resume.', flush=True)
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['init','pilot','check-pilot','detection','timing','status'])
    p.add_argument('--root', default='outputs/answer-boundary')
    p.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    p.add_argument('--datasets', nargs='+', choices=['SQuAD','race','nq_open','triviaqa'],
                   help='Optional detection-only subset; the plan still retains every required job')
    p.add_argument('--hours', type=float, default=18)
    p.add_argument('--hard-hours', type=float, default=20)
    p.add_argument('--workers', type=int, default=1, help='Fixed assignment count for separate servers')
    p.add_argument('--worker-index', type=int, default=0)
    p.add_argument('--wait-for-detection', action='store_true',
                   help='Timing only: wait for all detection parts, within the same total worker budget')
    args = p.parse_args()
    if not (0 < args.hours < args.hard_hours <= 22) or not 0 <= args.worker_index < args.workers:
        p.error('Require 0 < hours < hard-hours <= 22 and 0 <= worker-index < workers')
    if args.wait_for_detection and args.action != 'timing':
        p.error('--wait-for-detection is only valid for timing')
    if args.datasets and args.action != 'detection':
        p.error('--datasets is only valid for detection')
    # Fix CPU thread budgets for these new runs and record them in each runtime.
    # These values take effect before the child imports Torch/NumPy.
    os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                      TOKENIZERS_PARALLELISM='false')
    root = initialize(args.root)
    if args.action == 'init':
        print(f'Plans ready: {root}/plan.json (242 parts); {root}/pilot/plan.json (8 pilots)')
        return
    if args.action == 'status':
        from hide.parts import status
        status(root/'plan.json', [root])
        return
    if args.action == 'check-pilot':
        check_pilots(root, args.models)
        return
    start = time.monotonic()
    if args.action in {'pilot','detection'}:
        # Each model's four small pilots are checked before any full-cohort work.
        with queue_lock(root, shared=True):
            if not bounded_work(root, args.models, True, start, args.hours, args.hard_hours):
                return
            check_pilots(root, args.models)
            if args.action == 'detection':
                bounded_work(root, args.models, False, start, args.hours, args.hard_hours,
                             args.workers, args.worker_index, args.datasets)
        return
    if args.workers != 1 or args.worker_index != 0:
        p.error('Timing uses one worker on one GPU')
    # Both detector pilots must be complete; other model timing generations are
    # checked on every record by generation_fields and paired token-ID equality.
    with queue_lock(root, shared=True):
        if args.wait_for_detection and not wait_detection(root, start+args.hours*3600):
            return
        check_pilots(root, MODELS)
        elapsed = (time.monotonic()-start)/3600
        if elapsed >= args.hours:
            return
        with TimingMonitor(root):
            elapsed = (time.monotonic()-start)/3600
            if elapsed >= args.hours:
                return
            code = subprocess.call(['bash','scripts/run_priority.sh','--plan',str(root/'plan.json'),
                                    '--queue',str(root),'--kind','timing','--hours',str(args.hours-elapsed),
                                    '--hard-hours',str(args.hard_hours-elapsed)])
    if code:
        raise SystemExit(code)


if __name__ == '__main__':
    main()
