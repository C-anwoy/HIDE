#!/usr/bin/env python3
"""Replay one failed detection example in isolation with full keyword diagnostics."""
import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def diagnostic_arguments(path, destination):
    metadata = json.loads(path.with_suffix('.manifest.json').read_text())
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError('Run contains no records')
    failed = json.loads(lines[-1])
    if failed.get('status') != 'error':
        raise ValueError('Expected a final error record; do not alter the original run')
    args = metadata['arguments'].copy()
    if args['mode'] != 'detection':
        raise ValueError('This diagnostic supports detection failures only')
    ids = [str(value) for value in metadata['selected_ids']]
    index = ids.index(str(failed['id']))
    args['start'] = args.get('start', 0) + index
    args['stop'] = args['start'] + 1
    args['output'] = str(destination / 'replay.jsonl')
    args['resume'] = False
    args['time_budget_seconds'] = 1800
    return args, metadata, failed


def trace_keywords(original, destination):
    def traced(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except Exception:
            trace = traceback.format_exc()
            print(trace, file=sys.stderr, flush=True)
            (destination / 'keyword_traceback.txt').write_text(trace)
            try:
                bound = inspect.signature(original).bind(*args, **kwargs).arguments
                tokenizer = bound['tokenizer']
                details = {}
                for name in ('input_tokens', 'output_tokens'):
                    tokens = bound[name].tolist()
                    details[name] = tokens
                    details[name.replace('tokens', 'text')] = tokenizer.decode(tokens, skip_special_tokens=True)
                (destination / 'keyword_inputs.json').write_text(json.dumps(details, ensure_ascii=True, indent=2))
            except Exception:
                (destination / 'diagnostic_capture_error.txt').write_text(traceback.format_exc())
            raise
    return traced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path, help='Original failed JSONL; read only')
    parser.add_argument('--output', type=Path, help='New diagnostic directory; must not exist')
    parser.add_argument('--dry-run', action='store_true')
    options = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = (options.output or ROOT / 'outputs' / 'diagnostics' / stamp).resolve()
    path = options.run.resolve()
    args, metadata, failed = diagnostic_arguments(path, destination)
    from hide.parts import portable_sources
    from hide.provenance import source_files
    if portable_sources(metadata['source_sha256']) != portable_sources(source_files()):
        raise ValueError('Scientific source differs from the failed run; use its original code version')
    print(f'DIAGNOSTIC ONLY: example {failed["id"]}, positions {args["start"]}:{args["stop"]}', flush=True)
    print(f'Diagnostic folder: {destination}', flush=True)
    print('Original run and failure marker are unchanged. Do not merge diagnostic output into paper results.', flush=True)
    if options.dry_run:
        return
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, destination / 'diagnostic_launcher.py')
    (destination / 'request.json').write_text(json.dumps(dict(
        purpose='Isolated failure diagnostic; excluded from paper results', original_run=str(path),
        original_error=failed, arguments=args), indent=2))
    from hide import core, runner
    original = core.extract_keyword_representation
    core.extract_keyword_representation = trace_keywords(original, destination)
    previous_argv = sys.argv
    sys.argv = ['hide.runner']
    for name, value in args.items():
        flag = '--' + name.replace('_', '-')
        if isinstance(value, bool):
            if value:
                sys.argv.append(flag)
        elif value is not None:
            sys.argv.extend([flag, str(value)])
    try:
        runner.main()
    except BaseException:
        (destination / 'runner_traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        sys.argv = previous_argv
        core.extract_keyword_representation = original


if __name__ == '__main__':
    main()
