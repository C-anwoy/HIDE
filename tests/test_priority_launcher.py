"""Exercise the shell launcher's embedded scheduler without starting inference."""
from contextlib import redirect_stdout
import io
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = (Path(__file__).resolve().parents[1] / 'scripts/run_priority.sh').read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]


class PriorityLauncherTests(unittest.TestCase):
    def test_timing_diagnostic_preserves_failed_observation_and_original_exception(self):
        from contextlib import redirect_stderr
        spec = importlib.util.spec_from_file_location('timing_worker', Path('scripts/timing_worker.py'))
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        with tempfile.TemporaryDirectory() as directory:
            baseline = {'hostname': 'host', 'device': 'GPU-test, A100, 555, 300.00', 'plan_sha256': 'plan'}
            path = Path(directory)/'timing_device.json'
            path.write_text(json.dumps(baseline))
            error = ValueError('Timing worker GPU, host, driver or power limit changed')
            def original(queue, plan):
                identity = dict(baseline, device='GPU-test, A100, 555, 150.00')
                raise error
            stream = io.StringIO()
            with redirect_stderr(stream), self.assertRaises(ValueError) as caught:
                worker.diagnose_guard(original)(directory, {'sha256': 'plan'})
            self.assertIs(caught.exception, error)
            record = json.loads(stream.getvalue().split('=', 1)[1])
            self.assertEqual(record['observed_at_failed_check']['device'], 'GPU-test, A100, 555, 150.00')
            self.assertEqual(record['expected_file_after_check'], baseline)
            self.assertEqual(json.loads(path.read_text()), baseline)

    def test_detection_only_schedule_excludes_timing(self):
        tasks = [dict(id='qa', profile='qa', model='llama3-8b', dataset='nq_open', start=0),
                 dict(id='timing', profile='timing', model='gemma-2-27b', dataset='SQuAD', start=0)]
        output = io.StringIO()
        with patch('sys.argv', ['priority', '--kind', 'detection', '--dry-run']), \
             patch('hide.parts.load_plan', return_value={'suite': 'review', 'tasks': tasks}), \
             redirect_stdout(output), self.assertRaises(SystemExit) as caught:
            exec(compile(SCRIPT, 'run_priority.sh', 'exec'), {})
        self.assertEqual(caught.exception.code, 0)
        self.assertEqual(output.getvalue().splitlines(), ['qa'])

    def run_launcher(self, fail=False):
        tasks = [dict(id=f'{profile}-{model}-{dataset}', profile=profile, model=model,
                      dataset=dataset, start=0, stop=200)
                 for profile, model, dataset in [
                     ('qa', 'gemma-2-9b', 'triviaqa'),
                     ('timing', 'gemma-2-27b', 'SQuAD'),
                     ('qa', 'gemma-2-9b', 'nq_open'),
                     ('qa', 'llama3-8b', 'nq_open'),
                     ('qa', 'llama3-8b', 'SQuAD')]]
        calls, clock = [], [100.0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            class Child:
                def __init__(self, cmd, **kwargs):
                    calls.append(cmd)
                    self.task = cmd[cmd.index('--task') + 1]

                def wait(self):
                    clock[0] += 3600
                    if not fail:
                        (root / self.task).touch()
                    return 1 if fail else 0

            with patch('sys.argv', ['priority', '--queue', directory, '--hours', '2', '--hard-hours', '4']), \
                 patch('hide.parts.load_plan', return_value={'suite': 'review', 'tasks': tasks}), \
                 patch('hide.parts.part_path', side_effect=lambda queue, task: root / task['id']), \
                 patch('hide.export_results.inspect_run', return_value={'complete': True}), \
                 patch('subprocess.Popen', Child), patch('signal.signal'), \
                 patch('time.monotonic', side_effect=lambda: clock[0]), redirect_stdout(io.StringIO()):
                if fail:
                    with self.assertRaisesRegex(SystemExit, 'Worker failed'):
                        exec(compile(SCRIPT, 'run_priority.sh', 'exec'), {})
                else:
                    exec(compile(SCRIPT, 'run_priority.sh', 'exec'), {})
        return calls

    def test_priority_and_shared_budget(self):
        calls = self.run_launcher()
        self.assertEqual([cmd[cmd.index('--task')+1] for cmd in calls],
                         ['qa-llama3-8b-nq_open', 'qa-gemma-2-9b-nq_open'])
        self.assertEqual([float(cmd[cmd.index('--hours')+1]) for cmd in calls], [2, 1])
        self.assertEqual([float(cmd[cmd.index('--hard-hours')+1]) for cmd in calls], [4, 3])

    def test_failure_stops_without_retrying(self):
        self.assertEqual(len(self.run_launcher(fail=True)), 1)

    def timing_trial(self, failures):
        tasks = [dict(id='timing-3b', profile='timing', model='llama3-3b', dataset='SQuAD', start=0),
                 dict(id='timing-8b', profile='timing', model='llama3-8b', dataset='nq_open', start=0)]
        calls, clock = [], [0.0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = json.dumps({'hostname': 'host', 'device': 'GPU-test, A100, 555, 300.00', 'plan_sha256': 'plan'})
            (root/'timing_device.json').write_text(baseline)
            class Child:
                def __init__(self, cmd, stderr=None):
                    calls.append(cmd)
                    self.error = failures[len(calls)-1] if len(calls) <= len(failures) else ''
                    self.task = cmd[cmd.index('--task')+1]
                    if self.error:
                        stderr.write(self.error)
                def wait(self):
                    clock[0] += 1
                    if not self.error:
                        (root/self.task).touch()
                    return 1 if self.error else 0
            error = None
            with patch('sys.argv', ['priority', '--kind', 'timing', '--queue', directory]), \
                 patch('hide.parts.load_plan', return_value={'suite': 'review', 'tasks': tasks, 'sha256': 'plan'}), \
                 patch('hide.parts.part_path', side_effect=lambda queue, task: root/task['id']), \
                 patch('hide.export_results.inspect_run', return_value={'complete': True}), \
                 patch('subprocess.Popen', Child), patch('signal.signal'), \
                 patch('subprocess.check_output', return_value='GPU-test, A100, 555, 150.00\n'), \
                 patch('platform.node', return_value='host'), patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-test'), \
                 patch('time.monotonic', side_effect=lambda: clock[0]), \
                 patch('time.sleep', side_effect=lambda seconds: clock.__setitem__(0, clock[0]+seconds)), \
                 redirect_stdout(io.StringIO()):
                try:
                    exec(compile(SCRIPT, 'run_priority.sh', 'exec'), {})
                except SystemExit as exc:
                    error = str(exc)
            self.assertEqual((root/'timing_device.json').read_text(), baseline)
            reports = [json.loads(path.read_text()) for path in root.glob('sessions/timing_guard_*/*.json')]
            complete = [task['id'] for task in tasks if (root/task['id']).exists()]
        return calls, reports, complete, error

    def test_transient_timing_guard_retry_is_logged_and_budget_keeps_decreasing(self):
        calls, reports, complete, error = self.timing_trial([
            'ValueError: Timing worker GPU, host, driver or power limit changed; keep one timing platform\n'])
        self.assertIsNone(error)
        self.assertEqual([cmd[cmd.index('--task')+1] for cmd in calls], ['timing-3b', 'timing-3b', 'timing-8b'])
        self.assertEqual(len(reports), 1)
        self.assertIn('device', reports[0]['differences'])
        self.assertEqual(complete, ['timing-3b', 'timing-8b'])
        deadlines = [float(cmd[cmd.index('--hard-hours')+1]) for cmd in calls]
        self.assertTrue(all(a > b for a, b in zip(deadlines, deadlines[1:])))

    def test_persistent_timing_guard_stops_after_six_attempts(self):
        calls, reports, complete, error = self.timing_trial([
            'ValueError: Timing worker GPU, host, driver or power limit changed\n'] * 6)
        self.assertEqual(len(calls), 6)
        self.assertEqual(len(reports), 6)
        self.assertEqual(complete, [])
        self.assertIn('Worker failed', error)

    def test_timing_model_error_is_not_retried(self):
        calls, reports, complete, error = self.timing_trial(['RuntimeError: Part failed (1)\n'])
        self.assertEqual(len(calls), 1)
        self.assertEqual(reports, [])
        self.assertEqual(complete, [])
        self.assertIn('Worker failed', error)
