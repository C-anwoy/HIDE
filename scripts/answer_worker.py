#!/usr/bin/env python3
"""Wait briefly for plan initialization without changing the experiment sources."""
from functools import wraps
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def wait_for_initialization(original, timeout=60, interval=1):
    @wraps(original)
    def initialize(root, detection_only=False):
        root = Path(root).resolve()
        paths = [root/'plan.json', root/'pilot/plan.json']
        messages = {f'Another process is writing {path}': path for path in paths}
        deadline = time.monotonic() + timeout
        announced = set()
        while True:
            try:
                return original(root, detection_only=detection_only)
            except RuntimeError as exc:
                path = messages.get(str(exc))
                if path is None or not isinstance(exc.__cause__, BlockingIOError):
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    print(f'Plan startup lock still busy after {timeout}s: '
                          f'{path.with_suffix(".lock")}. Inspect its owner; do not delete it.',
                          file=sys.stderr, flush=True)
                    raise
                if path not in announced:
                    print(f'Waiting up to {timeout}s for plan startup lock: '
                          f'{path.with_suffix(".lock")}', file=sys.stderr, flush=True)
                    announced.add(path)
                time.sleep(min(interval, remaining))
    return initialize


def main():
    from hide import answer_runs
    original = answer_runs.initialize
    answer_runs.initialize = wait_for_initialization(original)
    try:
        answer_runs.main()
    finally:
        answer_runs.initialize = original


if __name__ == '__main__':
    main()
