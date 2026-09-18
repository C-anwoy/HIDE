import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from hide import answer_runs, parts
from hide.provenance import run_lock
from scripts.answer_subset import run, selection, selected_tasks, status


class AnswerSubsetTests(unittest.TestCase):
    def test_tmux_launches_two_persistent_shells_with_model_commands(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary = root/'tmux'
            log = root/'calls.jsonl'
            binary.write_text('#!'+sys.executable+'\nimport os,sys,json\n'
                'if sys.argv[1]=="has-session": sys.exit(1)\n'
                'with open(os.environ["HIDE_SUBSET_TMUX_LOG"],"a") as f: '
                'f.write(json.dumps(sys.argv[1:])+"\\n")\n')
            binary.chmod(0o755)
            env = dict(os.environ, HIDE_SUBSET_TMUX_LOG=str(log),
                       PATH=str(root)+os.pathsep+str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'])
            subprocess.run(['bash', 'scripts/start_subset_tmux.sh'], env=env,
                           capture_output=True, text=True, check=True)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len([c for c in calls if c[0] == 'new-session']), 2)
            commands = [c[-1] for c in calls if c[0] == 'send-keys' and '-l' in c]
            self.assertEqual(len(commands), 2)
            self.assertIn('llama3-8b', commands[0])
            self.assertIn('gemma-2-9b', commands[1])

    def test_fixed_subset_exact_coverage_and_factuality_priority(self):
        plan = parts.make_plan('answer-detection')
        tasks = selected_tasks(plan)
        self.assertEqual(len(tasks), 80)
        self.assertEqual(sum(t['stop']-t['start'] for t in tasks), 16000)
        self.assertTrue(all(t['dataset'] in ['nq_open', 'triviaqa'] for t in tasks[:40]))
        self.assertTrue(all(t['stop'] <= 2000 for t in tasks))
        self.assertEqual({t['id'] for t in tasks},
                         {t['id'] for t in plan['tasks'] if t['start'] < 2000})
        with self.assertRaises(ValueError):
            selected_tasks(parts.make_plan('answer-detection', size=300))

    def test_selection_preserves_parent_plan_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as td:
            root = answer_runs.initialize(td, detection_only=True)
            before = (root/'plan.json').read_bytes()
            first = selection(root)
            self.assertEqual(selection(root), first)
            self.assertEqual((root/'plan.json').read_bytes(), before)
            self.assertTrue((root/'subsets/answer-2000.json').exists())

    def test_worker_filters_only_detection_keeps_pilots_and_restores_functions(self):
        plan = parts.make_plan('answer-detection')
        pilot = parts.make_plan('answer-pilots')
        choose = parts.choose_tasks
        init = answer_runs.initialize
        def main():
            self.assertEqual(parts.choose_tasks(plan), selected_tasks(plan))
            self.assertEqual(parts.choose_tasks(pilot), choose(pilot))
            raise ValueError('worker failure')
        with patch('hide.answer_runs.main', side_effect=main), self.assertRaises(ValueError):
            run(Path('/tmp/unused'), 'gemma-2-9b', selected_tasks(plan), 8, 9)
        self.assertIs(parts.choose_tasks, choose)
        self.assertIs(answer_runs.initialize, init)

    def test_status_reports_completed_missing_and_live_parts(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            tasks = selected_tasks(parts.make_plan('answer-detection'))[:3]
            path = parts.part_path(root, tasks[0])
            path.parent.mkdir(parents=True)
            path.touch()
            with run_lock(root/'claims'/f'{tasks[2]["id"]}.claim'), \
                 patch('scripts.answer_subset.inspect_run', return_value={'complete': True}):
                result = status(root, tasks)
            self.assertEqual(result, dict(total=3, complete=1, missing=1, running=1, partial=0, failed=0))
