"""Resolve named experiments into explicit, logged, resumable commands."""
import argparse
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path.cwd().resolve()
CONFIG = Path(__file__).resolve().parent/'config'
DATASETS = ['SQuAD', 'race', 'nq_open', 'triviaqa']


def catalog():
    return json.loads((CONFIG/'models.json').read_text()), json.loads((CONFIG/'experiments.json').read_text())


def suite_jobs(suite, stack=()):
    _, experiments = catalog()
    if suite in stack:
        raise ValueError('Recursive suite definition')
    jobs = []
    for group in experiments['suites'][suite]:
        if 'include' in group:
            jobs.extend(suite_jobs(group['include'], (*stack, suite)))
        else:
            jobs.extend(itertools.product(group['profiles'], group['models'], group['datasets']))
    return list(dict.fromkeys(jobs))


def profile_arguments(profile):
    _, experiments = catalog()
    return dict(mode='detection', samples=0, seed=42, keywords=20, max_new_tokens=256,
                warmup=10, repeats=3, decoding='greedy', temperature=1.0, top_p=1.0,
                multipass_samples=0, ablations=False, **{}) | experiments['profiles'][profile]


def command(profile, model, dataset, output_root=None):
    models, experiments = catalog()
    if profile not in experiments['profiles'] or model not in models or dataset not in DATASETS:
        raise ValueError(f'Unknown experiment: {profile}/{model}/{dataset}')
    model_root = Path(os.environ.get('HIDE_MODEL_ROOT', 'checkpoints')).expanduser().resolve()
    data_root = Path(os.environ.get('HIDE_DATA_ROOT', ROOT/'data')).expanduser().resolve()
    result_root = Path(output_root or os.environ.get('HIDE_RESULTS_ROOT', ROOT/'outputs/final')).expanduser().resolve()
    output = result_root/profile/f'{model}_{dataset}.jsonl'
    args = profile_arguments(profile)
    args.update(profile=profile, model_name=model, model_path=str(model_root/models[model]['checkpoint']),
                data_root=str(data_root), dataset=dataset, layer=models[model]['layer'],
                keyword_model=str(model_root/'all-MiniLM-L6-v2'), judge_model=str(model_root/'nli-roberta-large'),
                device=os.environ.get('HIDE_DEVICE', 'cuda:0'), keyword_device='cpu', judge_device=os.environ.get('HIDE_DEVICE', 'cuda:0'),
                dtype=os.environ.get('HIDE_DTYPE', 'bfloat16'), attention_backend='eager', output=str(output), resume=True)
    cmd = [sys.executable, '-u', '-m', 'hide.runner']
    for name, value in args.items():
        flag = '--'+name.replace('_', '-')
        if isinstance(value, bool):
            if value: cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd, output


def execute(profile, model, dataset, dry_run=False):
    cmd, output = command(profile, model, dataset)
    print(shlex.join(cmd), flush=True)
    if dry_run:
        return
    # Missing checkpoints fail before loading datasets or allocating GPU memory.
    for flag in ['--model-path', '--keyword-model'] + ([] if profile == 'timing' else ['--judge-model']):
        location = Path(cmd[cmd.index(flag)+1])
        if not location.is_dir():
            raise FileNotFoundError(f'{flag}: {location}; configure HIDE_MODEL_ROOT/hide/config/models.json')
    logs = output.parent.parent/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    stem = f'{profile}_{model}_{dataset}'
    package_file = logs/f'{stem}.packages.txt'
    if not package_file.exists():
        with package_file.open('w') as stream:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
    environment = os.environ.copy()
    environment.setdefault('TOKENIZERS_PARALLELISM', 'false')
    with (logs/f'{stem}.log').open('a', buffering=1) as log:
        log.write('\nCOMMAND: '+shlex.join(cmd)+'\n')
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=environment)
        try:
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
            code = proc.wait()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, cmd)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('profile', nargs='?')
    p.add_argument('model', nargs='?')
    p.add_argument('dataset', nargs='?')
    p.add_argument('--suite', choices=list(catalog()[1]['suites']))
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.suite:
        if args.profile or args.model or args.dataset:
            p.error('Use --suite or a profile/model/dataset triple')
        jobs = suite_jobs(args.suite)
    elif not all([args.profile, args.model, args.dataset]):
        p.error('Provide PROFILE MODEL DATASET or --suite NAME')
    else:
        jobs = [(args.profile, args.model, args.dataset)]
    for profile, model, dataset in jobs:
        execute(profile, model, dataset, args.dry_run)


if __name__ == '__main__':
    main()
