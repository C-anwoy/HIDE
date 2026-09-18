import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from hide.answer_runs import initialize
from hide.provenance import run_lock
from scripts.answer_worker import main, wait_for_initialization


class AnswerStartupTests(unittest.TestCase):
    def test_real_process_waits_for_each_plan_lock_then_initializes(self):
        for relative in ['plan.json', 'pilot/plan.json']:
            with self.subTest(lock=relative), tempfile.TemporaryDirectory() as td:
                process = None
                try:
                    with run_lock(Path(td)/relative):
                        process = subprocess.Popen(
                            [sys.executable, '-u', 'scripts/answer_worker.py', 'init',
                             '--root', td, '--detection-only'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        # Synchronize on the actual contention, not an arbitrary sleep.
                        import select
                        ready, _, _ = select.select([process.stderr], [], [], 20)
                        self.assertTrue(ready, 'Child did not reach startup lock')
                        message = process.stderr.readline()
                        self.assertIn('Waiting up to 60s', message)
                        self.assertIn(str((Path(td)/relative).with_suffix('.lock')), message)
                        self.assertIsNone(process.poll())
                    out, err = process.communicate(timeout=20)
                    self.assertEqual(process.returncode, 0, out+err)
                    self.assertTrue((Path(td)/'plan.json').exists())
                    self.assertTrue((Path(td)/'pilot/plan.json').exists())
                finally:
                    if process is not None:
                        if process.poll() is None:
                            process.kill()
                        process.communicate()

    def test_timeout_keeps_live_lock_and_propagates_failure(self):
        with tempfile.TemporaryDirectory() as td, run_lock(Path(td)/'plan.json'):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(RuntimeError):
                wait_for_initialization(initialize, timeout=0)(td, detection_only=True)
            self.assertIn('do not delete', err.getvalue())
            with self.assertRaises(RuntimeError):
                with run_lock(Path(td)/'plan.json'):
                    self.fail('Live lock bypassed')

    def test_only_real_plan_contention_is_retried(self):
        errors = [ValueError('source fingerprint changed'),
                  RuntimeError('Another process is writing /tmp/test/plan.json'),
                  RuntimeError('Another process is writing /tmp/test/run.jsonl')]
        errors[-1].__cause__ = BlockingIOError()
        for error in errors:
            original = Mock(side_effect=error)
            with patch('scripts.answer_worker.time.sleep') as sleep:
                with self.assertRaises(type(error)):
                    wait_for_initialization(original)('/tmp/test')
                original.assert_called_once()
                sleep.assert_not_called()

    def test_wrapper_restores_initializer_on_failure(self):
        with patch('hide.answer_runs.main', side_effect=ValueError('failure')):
            with self.assertRaises(ValueError):
                main()
        from hide import answer_runs
        self.assertIs(answer_runs.initialize, initialize)
