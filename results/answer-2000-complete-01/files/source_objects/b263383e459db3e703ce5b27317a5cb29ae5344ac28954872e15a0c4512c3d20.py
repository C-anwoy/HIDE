"""Lossless, size-bounded Git export; verify/restore with only Python's standard library."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

CHUNK_BYTES = 16 * 1024 * 1024
DATASET_COUNTS = {'SQuAD': 5928, 'race': 3498, 'nq_open': 3610, 'triviaqa': 9960}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Unsafe archive path: {relative}')
    return path


def inspect_run(path):
    initial_stat = path.stat()
    manifest = path.with_suffix('.manifest.json')
    issues = []
    if not manifest.is_file():
        return {'source': str(path), 'complete': False, 'issues': ['Missing manifest']}
    meta = json.loads(manifest.read_text())
    args = meta['arguments']
    if meta.get('schema_version', 1) >= 2 and not path.with_suffix('.runtime.json').is_file():
        issues.append('Missing runtime metadata')
    repeats = args['repeats'] if args['mode'] == 'timing' else 1
    expected = {(str(i), rep) for i in meta['selected_ids'] for rep in range(repeats)}
    seen = set()
    good = set()
    n_error = 0
    variant_errors = 0
    variant_undefined = 0
    counts = {'no_output_state': 0, 'hit_generation_cap': 0}
    total_wall_s = 0.0
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                issues.append(f'Invalid JSON on line {number}; retained verbatim in export')
                continue
            key = (str(row['id']), int(row.get('repeat', 0)))
            if key in seen:
                issues.append(f'Duplicate example/repeat: {key}')
            seen.add(key)
            if row.get('status') == 'ok':
                good.add(key)
            else:
                n_error += 1
            for c in counts:
                counts[c] += int(bool(row.get(c, False)))
            total_wall_s += row.get('example_wall_s', 0.0)
            if meta.get('expected_ablation_names') and row.get('status') == 'ok':
                variants = row.get('ablations', [])
                names = [v.get('name') for v in variants]
                if sorted(names) != sorted(meta['expected_ablation_names']):
                    issues.append(f'Missing/duplicate ablation variants on line {number}')
                variant_errors += sum(v.get('status') == 'error' for v in variants)
                variant_undefined += sum(v.get('status') == 'undefined' for v in variants)
    missing = expected - good
    unexpected = seen - expected
    if missing:
        issues.append(f'{len(missing)} expected successful records missing')
    if unexpected:
        issues.append(f'{len(unexpected)} unexpected example/repeat keys')
    if n_error:
        issues.append(f'{n_error} error records')
    if variant_errors:
        issues.append(f'{variant_errors} unexpected ablation failures')
    file_digest = digest(path.read_bytes())
    final_stat = path.stat()
    if (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
        raise ValueError('A run changed during inspection; export between jobs or after stopping its writer')
    # New runs retain exact source snapshots. Verify them against their manifest hashes.
    for source, sha in meta.get('source_sha256', {}).items():
        p = Path(source)
        snapshot = path.with_suffix('.sources') / (p.parent.name+'__'+p.name)
        if not snapshot.is_file() or digest(snapshot.read_bytes()) != sha:
            issues.append(f'Missing or mismatched source snapshot: {p.name}')
    source_identity = sorted((Path(p).parent.name+'__'+Path(p).name, sha) for p, sha in meta.get('source_sha256', {}).items())
    return {'source': str(path), 'source_fingerprint': digest(json.dumps(source_identity).encode()) if source_identity else None,
            'cohort_sha256': meta.get('cohort_sha256'),
            'model': args['model_name'], 'dataset': args['dataset'],
            'observed_file_sha256': file_digest,
            'arguments': args, 'mode': args['mode'], 'samples_argument': args['samples'], 'repeats': repeats,
            'expected': len(expected), 'observed': len(seen), 'successful': len(good),
            'n_errors': n_error, 'ablation_errors': variant_errors, 'ablation_undefined': variant_undefined, **counts, 'sum_example_wall_s': total_wall_s,
            'complete': not issues, 'issues': issues,
            'missing_keys': sorted(missing), 'unexpected_keys': sorted(unexpected)}


def suite_issues(runs, suite='review'):
    from hide.launch import suite_jobs, profile_arguments, catalog
    models, _ = catalog()
    issues = []
    matched_dtypes = set()
    matched_sources = set()
    cohorts = {}
    for profile, model, dataset in suite_jobs(suite):
        expected = profile_arguments(profile)
        target_count = expected['samples'] or DATASET_COUNTS[dataset]
        expected_records = target_count * (expected['repeats'] if expected['mode'] == 'timing' else 1)
        matches = []
        for r in runs:
            args = r.get('arguments', {})
            if not (r.get('model') == model and r.get('dataset') == dataset
                    and r.get('expected') == expected_records and r.get('complete')):
                continue
            fields = dict(expected, layer=models[model]['layer'], attention_backend='eager')
            # Full comparison/ablation runs can supply the same default greedy target and proxies.
            if profile == 'qa':
                fields.pop('multipass_samples')
                fields.pop('ablations')
            if all(args.get(k) == v for k, v in fields.items()):
                matches.append(r)
        if not matches:
            issues.append(f'Missing complete {profile}: {model}/{dataset} ({expected_records} records; configured protocol)')
        else:
            matched_dtypes.update(r['arguments'].get('dtype') for r in matches)
            matched_sources.update(r['source_fingerprint'] for r in matches if r.get('source_fingerprint'))
            cohorts.setdefault((profile, dataset), set()).update(r['cohort_sha256'] for r in matches if r.get('cohort_sha256'))
    if len(matched_dtypes) > 1:
        issues.append('Mixed model dtypes in the requested suite; use a consistent configuration')
    if len(matched_sources) > 1:
        issues.append('Mixed code snapshots in the requested suite; audit the difference before combining runs')
    if any(len(values) > 1 for values in cohorts.values()):
        issues.append('Different prompt/reference cohorts for the same dataset/profile across models')
    return issues


def verify(bundle):
    bundle = Path(bundle).resolve()
    index = json.loads((bundle/'INDEX.json').read_text())
    if index.get('format') != 'hide-results-v1':
        raise ValueError('Unknown bundle format')
    for entry in index['files']:
        sha = hashlib.sha256()
        n_bytes = 0
        for part in entry['parts']:
            data = safe_path(bundle, part['path']).read_bytes()
            if digest(data) != part['sha256']:
                raise ValueError(f'Corrupt exported file: {part["path"]}')
            raw = gzip.decompress(data) if part['compression'] == 'gzip' else data
            sha.update(raw)
            n_bytes += len(raw)
        if sha.hexdigest() != entry['source_sha256'] or n_bytes != entry['source_bytes']:
            raise ValueError(f'Original-file checksum mismatch: {entry["source"]}')
    return index


def export(source, output, allow_incomplete=False, require_suite=False, suites=()):
    from contextlib import ExitStack
    from hide.provenance import run_lock, queue_lock
    with ExitStack() as stack:
        if (Path(source)/'plan.json').exists():
            stack.enter_context(queue_lock(source))
        for path in sorted(Path(source).rglob('*.jsonl')):
            stack.enter_context(run_lock(path))
        return _export(source, output, allow_incomplete, require_suite, suites)


def _export(source, output, allow_incomplete=False, require_suite=False, suites=()):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError('Choose a new export folder; existing snapshots are never overwritten')
    if output.is_relative_to(source):
        raise ValueError('Export must be outside the raw result directory')
    runs = []
    for path in sorted(source.rglob('*.jsonl')):
        run = inspect_run(path)
        run['source'] = str(path.relative_to(source))
        runs.append(run)
    issues = [f'{r["source"]}: {issue}' for r in runs for issue in r['issues']]
    if not runs:
        issues.append('No experiment JSONL files found')
    if (source/'plan.json').is_file():
        from hide.parts import load_plan, part_path
        plan = load_plan(source/'plan.json')
        complete_paths = {str(source/r['source']) for r in runs if r['complete']}
        pending = [t['id'] for t in plan['tasks'] if str(part_path(source,t)) not in complete_paths]
        if pending:
            issues.append(f'Work plan has {len(pending)} unfinished/missing parts in this queue')
    requested_suites = list(dict.fromkeys([*suites, *(['review'] if require_suite else [])]))
    for suite in requested_suites:
        issues += suite_issues(runs, suite)
    if issues and not allow_incomplete:
        raise ValueError('Export is incomplete; fix issues or use --allow-incomplete for a labeled checkpoint:\n'
                         + '\n'.join(issues[:25]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.hide-export-', dir=output.parent) as temporary:
        staging = Path(temporary)/'bundle'
        staging.mkdir()
        index = {'format': 'hide-results-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                 'complete': not issues, 'required_suite_checked': bool(requested_suites),
                 'suites_checked': requested_suites,
                 'issues': issues, 'runs': runs, 'files': []}
        for path in sorted(source.rglob('*')):
            if not path.is_file():
                continue
            if path.is_symlink():
                raise ValueError(f'Result files must not be symlinks: {path}')
            rel = path.relative_to(source).as_posix()
            sha = hashlib.sha256()
            n_bytes = 0
            parts = []
            # JSONL/logs and large files are losslessly compressed in bounded binary chunks.
            # A JSON line may span chunks; restore joins bytes exactly before parsing.
            compress = path.suffix in {'.jsonl', '.log'} or path.stat().st_size > CHUNK_BYTES
            with path.open('rb') as stream:
                number = 0
                while True:
                    raw = stream.read(CHUNK_BYTES)
                    if not raw and number:
                        break
                    sha.update(raw)
                    n_bytes += len(raw)
                    stored = gzip.compress(raw, mtime=0) if compress else raw
                    stored_rel = f'files/{rel}.part{number:04d}.gz' if compress else f'files/{rel}'
                    if any(part.endswith('.sources') for part in path.relative_to(source).parts[:-1]):
                        # Thousands of work parts use identical code snapshots: store each byte string once.
                        stored_rel = f'files/source_objects/{digest(stored)}{path.suffix}'
                    destination = safe_path(staging, stored_rel)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(stored)
                    parts.append({'path': stored_rel, 'compression': 'gzip' if compress else 'none',
                                  'bytes': len(stored), 'sha256': digest(stored)})
                    number += 1
                    if not raw or not compress:
                        break
            index['files'].append({'source': rel, 'source_sha256': sha.hexdigest(),
                                   'source_bytes': n_bytes, 'parts': parts})
            if path.suffix == '.jsonl':
                run = next(r for r in runs if r['source'] == rel)
                if run.get('observed_file_sha256') and run['observed_file_sha256'] != sha.hexdigest():
                    raise ValueError('A run changed while being exported; export between jobs or after stopping the writer')
        (staging/'INDEX.json').write_text(json.dumps(index, indent=2)+'\n')
        (staging/'README.md').write_text(
            '# HIDE experiment result snapshot\n\n'
            f"Complete: **{index['complete']}**. Suites checked: **{', '.join(requested_suites) or 'none'}**.\n\n"
            'Every file under the raw result directory is included losslessly. INDEX.json maps compressed '
            'chunks to original filenames and provides SHA-256 hashes, counts and completeness checks. '
            'Chunks are byte segments; restore before reading JSONL.\n\n'
            'Verify: `python -m hide.export_results --verify PATH_TO_THIS_FOLDER`\n\n'
            'Restore: `python -m hide.export_results --restore PATH_TO_THIS_FOLDER --output RESTORED_FOLDER`\n')
        verify(staging)
        shutil.move(str(staging), str(output))
    print(f'Exported {len(index["files"])} files; complete={index["complete"]}; {output}')
    return index


def restore(bundle, output):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    index = verify(bundle)
    if output.exists():
        raise FileExistsError('Restore to a new directory')
    output.mkdir(parents=True)
    for entry in index['files']:
        target = safe_path(output, entry['source'])
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('wb') as stream:
            for part in entry['parts']:
                data = safe_path(bundle, part['path']).read_bytes()
                stream.write(gzip.decompress(data) if part['compression'] == 'gzip' else data)
    print(f'Restored verified files to {output}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_mutually_exclusive_group(required=True)
    actions.add_argument('--source', help='Raw result directory to export')
    actions.add_argument('--verify', help='Verify an exported bundle')
    actions.add_argument('--restore', help='Restore an exported bundle')
    p.add_argument('--output')
    p.add_argument('--allow-incomplete', action='store_true')
    p.add_argument('--require-suite', action='store_true', help='Compatibility alias for --suite review')
    p.add_argument('--suite', action='append', default=[], choices=['pilots','review','consistency','ablations','decoding','timing',
                                                                 'answer-pilots','answer-review','answer-detection'])
    args = p.parse_args()
    if args.verify:
        index = verify(args.verify)
        print(f'Checksums verified; complete={index["complete"]}; files={len(index["files"])}')
    elif not args.output:
        p.error('--output is required for export/restore')
    elif args.restore:
        restore(args.restore, args.output)
    else:
        export(args.source, args.output, args.allow_incomplete, args.require_suite, args.suite)


if __name__ == '__main__':
    main()
