"""Exercise the shell launcher's embedded scheduler without starting inference."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = (Path(__file__).resolve().parents[1] / 'scripts/run_priority.sh').read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]


class PriorityLauncherTests(unittest.TestCase):
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
                def __init__(self, cmd):
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
