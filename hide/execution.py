"""Cooperative pauses between complete records, with no fabricated error labels."""
import math
import signal
import time


class RunPaused(Exception):
    pass


class RunControl:
    def __init__(self, seconds=0):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Time budget must be finite and nonnegative')
        self.deadline = time.monotonic() + seconds if seconds else None
        self.requested = False
        self.previous = {}

    def __enter__(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            self.previous[sig] = signal.signal(sig, self.request)
        return self

    def request(self, *_):
        self.requested = True

    def check(self):
        if self.requested or (self.deadline is not None and time.monotonic() >= self.deadline):
            raise RunPaused('stop requested or time budget reached')

    def __exit__(self, *_):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)
