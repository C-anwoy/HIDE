"""Plan small independent parts, run bounded workers, and merge complete cohorts."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

from hide.export_results import DATASET_COUNTS, inspect_run
from hide.launch import catalog, command, profile_arguments, suite_jobs
from hide.provenance import atomic_json, data_lock, file_sha256, queue_lock, run_lock, source_files


SUITES = ('review', 'full', 'consistency', 'ablations', 'decoding', 'timing')


def portable_sources(sources):
    return {Path(p).parent.name+'__'+Path(p).name: sha for p, sha in sources.items()}


def plan_digest(plan):
    value = {k: v for k, v in plan.items() if k != 'sha256'}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def make_plan(suite, size=200):
    if suite not in SUITES or size < 1:
        raise ValueError('Unknown suite or nonpositive part size')
    suites = ['consistency', 'ablations', 'decoding', 'timing'] if suite == 'full' else [suite]
    jobs = list(dict.fromkeys(job for name in suites for job in suite_jobs(name)))
    # The two mechanistic models supply the reviewer baseline table first.
    if suite == 'full':
        jobs.sort(key=lambda j: (0 if j[0] == 'comparison' and j[1] in ('llama3-8b','gemma-2-9b') else 1))
    tasks = []
    for profile, model, dataset in jobs:
        args = profile_arguments(profile)
        n = min(args['samples'] or DATASET_COUNTS[dataset], DATASET_COUNTS[dataset])
        # Keep a timing experiment intact: warmups/repeat ordering match the original protocol.
        step = n if args['mode'] == 'timing' else size
        for start in range(0, n, step):
            stop = min(start+step, n)
            tasks.append(dict(id=f'{profile}__{model}__{dataset}__{start:05d}-{stop:05d}',
                              profile=profile, model=model, dataset=dataset, start=start, stop=stop))
    plan = dict(format='hide-work-plan-v1', suite=suite, part_size=size,
                source_sha256=portable_sources(source_files()), tasks=tasks)
    plan['sha256'] = plan_digest(plan)
    return plan


def load_plan(path, check_source=False):
    plan = json.loads(Path(path).read_text())
    if plan.get('format') != 'hide-work-plan-v1' or plan.get('sha256') != plan_digest(plan):
        raise ValueError('Unknown or modified plan; regenerate before starting any runs')
    # Reconstruct the declared plan to reject overlaps, gaps, extra tasks and changed protocols.
    expected = make_plan(plan['suite'], plan['part_size'])
    if plan['tasks'] != expected['tasks']:
        raise ValueError('Plan tasks differ from the declared suite')
    if check_source and plan['source_sha256'] != expected['source_sha256']:
        raise ValueError('Code differs from the work plan. Use the same commit on all workers.')
    return plan


def part_path(queue, task):
    return Path(queue)/'runs'/task['id']/task['profile']/f'{task["model"]}_{task["dataset"]}.jsonl'


def choose_tasks(plan, kind='detection', index=0, workers=1):
    if workers < 1 or not 0 <= index < workers:
        raise ValueError('Require 0 <= worker-index < workers')
    tasks = [t for t in plan['tasks'] if (t['profile'] == 'timing') == (kind == 'timing')]
    return [t for i, t in enumerate(tasks) if i % workers == index]


def execute_child(cmd, log_path, soft_deadline, hard_deadline, events, stop_requested=lambda: False):
    """Soft SIGTERM finishes a record. Hard SIGKILL may require trailing-line recovery."""
    with log_path.open('a', buffering=1) as log:
        log.write('\nCOMMAND: '+json.dumps(cmd)+'\n')
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        signalled = False
        def stop(sig):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                pass  # The process finished between poll() and delivery.
        try:
            while proc.poll() is None:
                now = time.monotonic()
                if (now >= soft_deadline or stop_requested()) and not signalled:
                    stop(signal.SIGTERM)
                    events.append(dict(event='cooperative_stop', time=time.time()))
                    signalled = True
                if now >= hard_deadline:
                    stop(signal.SIGKILL)
                    events.append(dict(event='hard_stop', time=time.time()))
                    break
                time.sleep(.2)
            return proc.wait()
        finally:
            if proc.poll() is None:
                stop(signal.SIGKILL)
                proc.wait()


def timing_device(queue, plan):
    """Timing is tied to one physical GPU/host; also reject an already occupied GPU."""
    import platform
    device = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not device.startswith('GPU-') or ',' in device:
        raise ValueError('Timing requires CUDA_VISIBLE_DEVICES set to one full physical GPU UUID')
    if os.environ.get('HIDE_DEVICE', 'cuda:0') != 'cuda:0':
        raise ValueError('Use HIDE_DEVICE=cuda:0 after restricting GPU visibility')
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version,power.limit',
                                    '--format=csv,noheader,nounits'], text=True)
    devices = [line.strip() for line in rows.splitlines() if line.split(',')[0].strip() == device]
    if len(devices) != 1:
        raise ValueError('Configured timing UUID is not visible to nvidia-smi')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader,nounits'], text=True)
    if any(line.split(',')[0].strip() == device for line in processes.splitlines()):
        raise ValueError('Timing GPU has an active compute process. Wait for exclusive access.')
    identity = dict(hostname=platform.node(), device=devices[0], plan_sha256=plan['sha256'])
    path = Path(queue)/'timing_device.json'
    with run_lock(path):
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError('Timing worker GPU, host, driver or power limit changed; keep one timing platform')
        if not path.exists():
            atomic_json(path, identity)
    return identity


def work(args):
    with queue_lock(args.queue, shared=True):
        if args.kind == 'timing':
            with run_lock(Path(args.queue)/'timing-worker.claim'):
                return _work(args)
        return _work(args)


def _work(args):
    from hide.execution import RunControl
    started = time.monotonic()
    if not (math.isfinite(args.hours) and math.isfinite(args.hard_hours)
            and 0 < args.hours < args.hard_hours <= 22):
        raise ValueError('Require 0 < hours < hard-hours <= 22 (defaults: 18 and 20)')
    plan = load_plan(args.plan, check_source=True)
    queue = Path(args.queue).resolve()
    queue.mkdir(parents=True, exist_ok=True)
    with data_lock(queue, 'plan'):
        if (queue/'plan.json').exists() and json.loads((queue/'plan.json').read_text()) != plan:
            raise ValueError('Queue already belongs to a different plan')
        atomic_json(queue/'plan.json', plan)
    tasks = choose_tasks(plan, args.kind, args.worker_index, args.workers)
    if args.models:
        tasks = [t for t in tasks if t['model'] in args.models]
    if args.profiles:
        tasks = [t for t in tasks if t['profile'] in args.profiles]
    if args.task:
        tasks = [t for t in tasks if t['id'] == args.task]
        if not tasks:
            raise ValueError('Task not found in this worker assignment/kind')
    if args.max_parts < 0:
        raise ValueError('max-parts must be nonnegative')
    if args.kind == 'timing' and args.workers != 1:
        raise ValueError('Run all timing jobs with one worker on the same exclusive GPU')
    session = uuid.uuid4().hex
    sessions = queue/'sessions'; sessions.mkdir(exist_ok=True)
    record = dict(session=session, plan_sha256=plan['sha256'], arguments=vars(args), events=[],
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  packages=subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True))
    session_path = sessions/f'{session}.json'
    soft = started + args.hours*3600
    hard = started + args.hard_hours*3600
    completed = 0
    try:
        with RunControl() as control:
            for task in tasks:
                if time.monotonic() >= soft or control.requested or (args.max_parts and completed >= args.max_parts):
                    break
                claim = run_lock(queue/'claims'/f'{task["id"]}.claim')
                try:
                    claim.__enter__()
                except RuntimeError:
                    continue  # A different worker owns this part on the shared filesystem.
                try:
                    output = part_path(queue, task)
                    failure = output.with_suffix('.failed.json')
                    if failure.exists():
                        continue  # Inspect/recover explicitly; never endlessly retry a scientific error.
                    if output.exists():
                        report = inspect_run(output)
                        if report['complete']:
                            continue
                        invalid = [v for v in report['issues'] if 'expected successful records missing' not in v
                                   and v != 'Missing runtime metadata']
                        if invalid:
                            raise ValueError(f'{output}: needs explicit recovery: {invalid}')
                    if args.kind == 'timing':
                        record['timing_device'] = timing_device(queue, plan)
                    cmd, _ = command(task['profile'], task['model'], task['dataset'],
                                     output_root=queue/'runs'/task['id'])
                    cmd += ['--start', str(task['start']), '--stop', str(task['stop']),
                            '--time-budget-seconds', str(max(.001, soft-time.monotonic()))]
                    # Downloads/hash preparation happen inside the supervised runner budget.
                    output.parent.mkdir(parents=True, exist_ok=True)
                    log = output.with_suffix('.log')
                    record['events'].append(dict(event='start', task=task['id'], time=time.time(), command=cmd))
                    atomic_json(session_path, record)
                    print(f'RUN {task["id"]}\nLog: {log}', flush=True)
                    code = execute_child(cmd, log, soft, hard, record['events'], lambda: control.requested)
                    report = inspect_run(output) if output.exists() else {'complete': False}
                    record['events'].append(dict(event='finish', task=task['id'], time=time.time(),
                                                  exit_code=code, complete=report['complete']))
                    atomic_json(session_path, record)
                    if code not in (0,75,-signal.SIGKILL,-signal.SIGTERM):
                        atomic_json(failure, dict(task=task['id'], exit_code=code, log=str(log)))
                        raise RuntimeError(f'Part failed ({code}); inspect {log}')
                    if report['complete']:
                        completed += 1
                    elif code == 0:
                        raise RuntimeError(f'Child exited successfully with incomplete output: {output}')
                    else:
                        print('Paused. Rerun the same worker command to continue.', flush=True)
                        break
                finally:
                    claim.__exit__(None,None,None)
    finally:
        record['elapsed_s'] = time.monotonic()-started
        record['parts_completed_this_session'] = completed
        atomic_json(session_path, record)
    print(f'Worker stopped; {completed} parts completed this session. Check queue status for remaining work.')


def status(plan_path, queues):
    plan = load_plan(plan_path)
    counts = dict(complete=0, partial=0, missing=0, failed=0, duplicate=0, running=0)
    for task in plan['tasks']:
        files = [part_path(q,task) for q in queues if part_path(q,task).exists()]
        if len(files)>1:
            counts['duplicate'] += 1
        with ExitStack() as stack:
            try:
                for q in queues:
                    stack.enter_context(run_lock(Path(q)/'claims'/f'{task["id"]}.claim'))
                for path in files:
                    stack.enter_context(run_lock(path))
            except RuntimeError:
                counts['running'] += 1
                continue
            if any(part_path(q,task).with_suffix('.failed.json').exists() for q in queues):
                counts['failed'] += 1
            elif not files:
                counts['missing'] += 1
            elif any(inspect_run(p)['complete'] for p in files):
                counts['complete'] += 1
            else:
                counts['partial'] += 1
    print(json.dumps(dict(total=len(plan['tasks']), **counts), indent=2))
    return counts


def identity(meta):
    """Scientific identity, independent of absolute paths and physical accuracy GPU."""
    value = deepcopy(meta)
    args = value['arguments']
    for key in ('output','start','stop','model_path','keyword_model','judge_model','data_root',
                'device','judge_device','keyword_device'):
        args.pop(key, None)
    for key in ('selected_ids','cohort_sha256','partition','gpu','dataset_fingerprint'):
        value.pop(key, None)
    value['source_sha256'] = portable_sources(value.get('source_sha256', {}))
    value.get('checkpoint_config', {}).pop('_name_or_path', None)
    for checkpoint in value.get('checkpoint_identity', {}).values():
        checkpoint.pop('location', None)
        # Preserve timestamps unless full content hashes prove the copied weights identical.
        for item in checkpoint.get('files', []):
            if 'sha256' in item:
                item.pop('mtime_ns', None)
    return value


def merge(plan_path, queues, output_root, kind='all'):
    plan = load_plan(plan_path, check_source=True)
    tasks = [t for t in plan['tasks'] if kind=='all' or (t['profile']=='timing') == (kind=='timing')]
    if not tasks:
        raise ValueError('No tasks match the merge kind')
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError('Merge to a new directory; existing results are never overwritten')
    queues = list(dict.fromkeys(Path(q).resolve() for q in queues))
    if any(output_root.is_relative_to(q) for q in queues):
        raise ValueError('Merged output must be outside input queues')
    selected = []
    with ExitStack() as locks:
        for queue in sorted(queues):
            locks.enter_context(queue_lock(queue))
        for task in tasks:
            copies = [part_path(q,task) for q in queues if part_path(q,task).exists()]
            if len(copies) != 1:
                raise ValueError(f'{task["id"]}: expected exactly one copy, found {len(copies)}')
            path = copies[0]
            locks.enter_context(run_lock(path))
            report = inspect_run(path)
            if not report['complete']:
                raise ValueError(f'Incomplete {path}: {report["issues"]}')
            meta = json.loads(path.with_suffix('.manifest.json').read_text())
            part = meta.get('partition', {})
            if part.get('start') != task['start'] or part.get('stop') != task['stop']:
                raise ValueError(f'Partition differs from plan: {path}')
            if portable_sources(meta['source_sha256']) != plan['source_sha256']:
                raise ValueError(f'Code differs from plan: {path}')
            expected_args = profile_arguments(task['profile'])
            expected_args.update(profile=task['profile'], model_name=task['model'], dataset=task['dataset'],
                                 layer=catalog()[0][task['model']]['layer'], attention_backend='eager')
            if any(meta['arguments'].get(k) != v for k,v in expected_args.items()):
                raise ValueError(f'Run protocol differs from plan: {path}')
            n = expected_args['samples'] or DATASET_COUNTS[task['dataset']]
            parent = part.get('parent_selected_ids', [])
            if len(parent) != n or len(set(parent)) != n or meta['selected_ids'] != parent[task['start']:task['stop']]:
                raise ValueError(f'Invalid parent cohort or partition IDs: {path}')
            selected.append((task,path,meta))
        timing_platforms = set()
        timing_settings = set()
        for task,path,meta in selected:
            if task['profile']=='timing':
                device_record = path.parents[3]/'timing_device.json'
                if not device_record.is_file():
                    raise ValueError('Timing part lacks the worker platform record')
                settings = json.loads(device_record.read_text())
                timing_settings.add(json.dumps(settings, sort_keys=True))
                runtime = json.loads(path.with_suffix('.runtime.json').read_text())
                timing_platforms.add((runtime.get('hostname'),runtime.get('gpu_uuid') or runtime.get('cuda_visible_devices')))
        if len(timing_platforms)>1 or len(timing_settings)>1:
            raise ValueError('Timing results span physical GPUs/hosts; measure scaling on one platform')
        if any(not host or not dev for host,dev in timing_platforms):
            raise ValueError('Missing physical timing GPU identity')
        output_root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.hide-merge-', dir=output_root.parent) as tmp:
            staging=Path(tmp)/'merged'; staging.mkdir()
            groups={}
            for task,path,meta in selected:
                groups.setdefault((task['profile'],task['model'],task['dataset']), []).append((task,path,meta))
            provenance=[]
            for (profile,model,dataset), group in groups.items():
                group.sort(key=lambda item:item[0]['start'])
                first=group[0][2]; parent=first['partition']['parent_selected_ids']
                expected_identity=identity(first)
                target=staging/profile/f'{model}_{dataset}.jsonl'; target.parent.mkdir(exist_ok=True)
                emitted=[]; runtimes=[]
                observed_parent_hash=hashlib.sha256()
                with target.open('wb') as stream:
                    for task,path,meta in group:
                        if identity(meta)!=expected_identity or meta['partition']['parent_selected_ids']!=parent or \
                           meta['partition']['parent_cohort_sha256']!=first['partition']['parent_cohort_sha256']:
                            raise ValueError(f'Incompatible parts: {path}; check versions/checkpoints/cohorts')
                        rows={}
                        for line in path.read_bytes().splitlines(keepends=True):
                            row=json.loads(line); rows[(str(row['id']),row.get('repeat',0))]=line
                        repeats=meta['arguments']['repeats'] if profile=='timing' else 1
                        for example_id in meta['selected_ids']:
                            example=json.loads(rows[(example_id,0)])
                            observed_parent_hash.update(json.dumps([example_id,example['prompt'],example['answer']],
                                                                   ensure_ascii=False).encode())
                            for rep in range(repeats):
                                raw=rows[(example_id,rep)]
                                stream.write(raw if raw.endswith(b'\n') else raw+b'\n')
                            emitted.append(example_id)
                        saved=staging/'provenance'/'parts'/task['id']
                        # Original bytes, metadata, source snapshots, errors/recovery history and logs.
                        shutil.copytree(path.parents[1],saved)
                        for original in list(saved.rglob('*.jsonl')):
                            original.rename(original.with_suffix('.jsonl.bak'))
                        runtimes.append(dict(part=task['id'], metadata=meta,
                            runtime=json.loads(path.with_suffix('.runtime.json').read_text())))
                        provenance.append(dict(part=task['id'], original=str(path),sha256=file_sha256(path)))
                if emitted!=parent:
                    raise ValueError('Merged cohort has gaps, overlap or wrong order')
                if observed_parent_hash.hexdigest()!=first['partition']['parent_cohort_sha256']:
                    raise ValueError('Merged prompt/reference bytes differ from the declared parent cohort')
                merged=deepcopy(first); partition=merged.pop('partition')
                merged['selected_ids']=parent
                merged['cohort_sha256']=partition['parent_cohort_sha256']
                merged['arguments'].update(start=0,stop=None,output=str(output_root/profile/target.name))
                merged['gpu']='; '.join(sorted({item[2]['gpu'] for item in group}))
                merged['merged_parts']=[item[0]['id'] for item in group]
                atomic_json(target.with_suffix('.manifest.json'),merged)
                runtime=deepcopy(runtimes[0]['runtime']); runtime['part_executions']=runtimes
                atomic_json(target.with_suffix('.runtime.json'),runtime)
                shutil.copytree(group[0][1].with_suffix('.sources'),target.with_suffix('.sources'))
                if not inspect_run(target)['complete']:
                    raise ValueError(f'Merged run failed verification: {target}')
            for i,queue in enumerate(queues):
                if (queue/'sessions').is_dir():
                    shutil.copytree(queue/'sessions',staging/'provenance'/f'worker_sessions_{i}')
                if (queue/'timing_device.json').is_file():
                    shutil.copy2(queue/'timing_device.json',staging/'provenance'/f'timing_device_{i}.json')
            atomic_json(staging/'work_plan.json',plan)
            atomic_json(staging/'merge_manifest.json',dict(kind=kind, plan_sha256=plan['sha256'], parts=provenance,
                note='Accuracy GPU identities are retained per part; heterogeneous hardware may change floating-point generation.'))
            shutil.move(str(staging),str(output_root))
    print(f'Verified merge: {len(selected)} parts -> {output_root}')


def retry(queue, task_id):
    from hide.recover import recover
    queue = Path(queue)
    plan = load_plan(queue/'plan.json', check_source=True)
    matches = [t for t in plan['tasks'] if t['id']==task_id]
    if len(matches)!=1:
        raise ValueError('Unknown task ID')
    path=part_path(queue,matches[0])
    with queue_lock(queue):
        if path.exists():
            recover(path,retry_error=True)
        marker=path.with_suffix('.failed.json')
        if marker.exists():
            marker.rename(path.with_suffix('.failed.history.'+uuid.uuid4().hex+'.json'))
    print('Retry enabled after preserving failure history. Rerun the worker after fixing the cause.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action', required=True)
    create=sub.add_parser('plan'); create.add_argument('--suite',choices=SUITES,required=True)
    create.add_argument('--part-size',type=int,default=200); create.add_argument('--output',required=True)
    worker=sub.add_parser('work'); worker.add_argument('--plan',required=True); worker.add_argument('--queue',required=True)
    worker.add_argument('--kind',choices=['detection','timing'],default='detection')
    worker.add_argument('--worker-index',type=int,default=0); worker.add_argument('--workers',type=int,default=1)
    worker.add_argument('--models', nargs='+', choices=list(catalog()[0]))
    worker.add_argument('--profiles', nargs='+', choices=list(catalog()[1]['profiles']))
    worker.add_argument('--task', help='Run one exact task ID from the plan')
    worker.add_argument('--max-parts', type=int, default=0, help='Stop after this many completed parts; 0 means no count limit')
    worker.add_argument('--hours',type=float,default=18); worker.add_argument('--hard-hours',type=float,default=20)
    for name in ('status','merge'):
        action=sub.add_parser(name); action.add_argument('--plan',required=True)
        action.add_argument('--queues',nargs='+',required=True)
        if name=='merge':
            action.add_argument('--output',required=True)
            action.add_argument('--kind',choices=['all','detection','timing'],default='all')
    recover=sub.add_parser('retry'); recover.add_argument('--queue',required=True); recover.add_argument('--task',required=True)
    args=p.parse_args()
    if args.action=='retry':
        retry(args.queue,args.task)
    elif args.action=='plan':
        if Path(args.output).exists():
            raise FileExistsError('Plan already exists; reuse it or choose a new output')
        plan=make_plan(args.suite,args.part_size); atomic_json(args.output,plan)
        table = Path(args.output).with_suffix('.tsv')
        table.write_text('task_id\tprofile\tmodel\tdataset\tstart\tstop\texamples\n' + ''.join(
            f"{t['id']}\t{t['profile']}\t{t['model']}\t{t['dataset']}\t{t['start']}\t{t['stop']}\t{t['stop']-t['start']}\n"
            for t in plan['tasks']))
        print(f'{len(plan["tasks"])} independent parts; plan={args.output}; SHA256={plan["sha256"]}; table={table}')
    elif args.action=='work': work(args)
    elif args.action=='status': status(args.plan,args.queues)
    else: merge(args.plan,args.queues,args.output,args.kind)


if __name__=='__main__':
    main()
