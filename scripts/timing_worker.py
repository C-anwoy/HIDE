#!/usr/bin/env python3
"""Run the unchanged timing worker with diagnostics from the exact failing guard frame."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def diagnose_guard(original):
    def checked(queue, plan):
        try:
            return original(queue, plan)
        except ValueError as exc:
            frame = exc.__traceback__
            observed = None
            while frame is not None:
                if frame.tb_frame.f_code is original.__code__:
                    observed = frame.tb_frame.f_locals.get('identity')
                frame = frame.tb_next
            if observed is not None:
                try:
                    expected = json.loads((Path(queue)/'timing_device.json').read_text())
                    report = {'observed_at_failed_check': observed, 'expected_file_after_check': expected,
                              'differences': {key: {'expected': expected.get(key), 'observed': observed.get(key)}
                                              for key in set(expected) | set(observed)
                                              if expected.get(key) != observed.get(key)}}
                    print('TIMING_GUARD_DIAGNOSTICS=' + json.dumps(report), file=sys.stderr, flush=True)
                except Exception as diagnostic_error:
                    print(f'Timing diagnostic capture failed: {diagnostic_error}', file=sys.stderr, flush=True)
            raise
    return checked


def main():
    from hide import parts
    original = parts.timing_device
    old_argv = sys.argv
    parts.timing_device = diagnose_guard(original)
    sys.argv = ['hide.parts', 'work', *old_argv[1:]]
    try:
        parts.main()
    finally:
        parts.timing_device = original
        sys.argv = old_argv


if __name__ == '__main__':
    main()
