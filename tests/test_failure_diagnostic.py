import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('failure_diagnostic', Path(__file__).resolve().parents[1] / 'scripts/diagnose_failure.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class FailureDiagnosticTests(unittest.TestCase):
    def test_replay_uses_failed_cohort_position_without_modifying_original(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / 'original.jsonl'
            raw = json.dumps({'id': 'a', 'status': 'ok'}) + '\n' + json.dumps({'id': '2720', 'status': 'error'}) + '\n'
            run.write_text(raw)
            metadata = {'selected_ids': ['a', '2720', 'c'], 'arguments': {
                'mode': 'detection', 'start': 3400, 'stop': 3600, 'output': str(run), 'seed': 42}}
            manifest = run.with_suffix('.manifest.json')
            manifest.write_text(json.dumps(metadata))
            args, _, _ = diagnostic.diagnostic_arguments(run, root / 'diagnostic')
            self.assertEqual((args['start'], args['stop']), (3401, 3402))
            self.assertEqual(args['seed'], 42)
            self.assertFalse(args['resume'])
            self.assertNotEqual(args['output'], str(run))
            self.assertEqual(run.read_text(), raw)
            self.assertEqual(json.loads(manifest.read_text()), metadata)

    def test_exception_chain_and_inputs_are_retained_without_recovery(self):
        class Tokens(list):
            def tolist(self):
                return list(self)

        class Tokenizer:
            def decode(self, tokens, **kwargs):
                return '!!!'

        def extraction(X, Y, tokenizer, input_tokens, output_tokens, **kwargs):
            try:
                raise ValueError('empty vocabulary')
            except ValueError as exc:
                raise RuntimeError('Output keyword extraction failed') from exc

        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()):
            root = Path(folder)
            traced = diagnostic.trace_keywords(extraction, root)
            with self.assertRaisesRegex(RuntimeError, 'Output keyword extraction failed'):
                traced(None, None, Tokenizer(), Tokens([1]), Tokens([2]))
            trace = (root / 'keyword_traceback.txt').read_text()
            self.assertIn('ValueError: empty vocabulary', trace)
            self.assertIn('RuntimeError: Output keyword extraction failed', trace)
            self.assertEqual(json.loads((root / 'keyword_inputs.json').read_text())['output_text'], '!!!')
