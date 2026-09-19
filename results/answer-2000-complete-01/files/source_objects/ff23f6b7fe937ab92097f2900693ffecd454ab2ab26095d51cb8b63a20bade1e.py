"""Analyze each experiment profile separately; never pool decoding or timing protocols."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from hide.export_results import inspect_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default=os.environ.get('HIDE_RESULTS_ROOT', 'outputs/final'))
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--bootstrap', type=int, default=2000)
    args = p.parse_args()
    root = Path(args.source).resolve()
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    import tempfile
    env.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'hide-matplotlib-cache'))
    def run(module, *params):
        subprocess.run([sys.executable, '-m', 'hide.'+module, *map(str, params)], env=env, check=True)
    run('summarize_runs', '--source', root)
    files = sorted(root.glob('*/*.jsonl'))
    if not files:
        raise ValueError('No run files: expected ROOT/PROFILE/MODEL_DATASET.jsonl')
    inspected = [inspect_run(path) for path in files]
    if not args.allow_partial and any(not r['complete'] for r in inspected):
        raise ValueError('Some runs are incomplete; finish them or explicitly use --allow-partial')
    analysis = root/'analysis'
    from hide.provenance import source_files, atomic_json
    sources = source_files()
    snapshots = analysis/'sources'; snapshots.mkdir(exist_ok=True)
    for source in sources:
        path = Path(source)
        (snapshots/(path.parent.name+'__'+path.name)).write_bytes(path.read_bytes())
    provenance = dict(partial_analysis=args.allow_partial, analysis_source_sha256=sources,
                      sources=[dict(path=str(p.relative_to(root)), sha256=r.get('observed_file_sha256'),
                                    complete=r['complete'], arguments=r.get('arguments')) for p, r in zip(files, inspected)])
    (analysis/'analysis_manifest.json').write_text(json.dumps(provenance, indent=2)+'\n')
    for profile in sorted({p.parent.name for p in files}):
        selected = [p for p in files if p.parent.name == profile]
        if profile in {'pilot', 'answer-pilot'}:
            continue  # Operational checks must not appear as scientific evaluation tables.
        records = [r for p, r in zip(files, inspected) if p in selected]
        source_versions = {r['source_fingerprint'] for r in records if r.get('source_fingerprint')}
        if len(source_versions) > 1:
            raise ValueError(f'Mixed code versions in {profile}; analyze in separate output roots')
        for ds in {r.get('dataset') for r in records}:
            cohorts = {r['cohort_sha256'] for r in records if r.get('dataset') == ds and r.get('cohort_sha256')}
            if len(cohorts) > 1:
                raise ValueError(f'Different prompt/reference cohorts in {profile}/{ds}')
        out = analysis/profile
        modes = {r.get('mode') for r in records}
        if len(modes) != 1:
            raise ValueError(f'Mixed modes in profile directory: {profile}')
        if modes == {'timing'}:
            run('summarize_timing', *selected, '--output-dir', out)
            continue
        run('evaluate', *selected, '--output-dir', out, '--bootstrap', args.bootstrap)
        run('plot_mechanistic', *selected, '--output-dir', out/'mechanistic')
        if any(r.get('arguments', {}).get('ablations') for r in records):
            run('analyze_ablations', *selected, '--output-dir', out/'ablations')
        if profile in {'qa', 'answer-qa', 'comparison', 'ablations'}:
            for lo, hi in [(0,1),(2,4),(5,9),(10,20)]:
                run('evaluate', *selected, '--n-eff-bin', lo, hi, '--bootstrap', args.bootstrap,
                    '--output-dir', out/f'count_{lo}_{hi}')
    print(analysis)


if __name__ == '__main__': main()
