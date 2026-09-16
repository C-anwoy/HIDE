import json
from pathlib import Path
import tempfile
import unittest
from hide.recover import recover
from hide.export_results import export
from hide.provenance import run_lock


class RecoveryTests(unittest.TestCase):
    def test_truncated_tail_is_backed_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run.jsonl'
            raw=b'{"id":"a","status":"ok"}\n{"id":"b"'
            path.write_bytes(raw)
            history=recover(path)
            self.assertEqual((history/'original.jsonl.bak').read_bytes(),raw)
            self.assertEqual(path.read_bytes(),raw.splitlines(keepends=True)[0])
            self.assertIsNone(recover(path))

    def test_errors_require_explicit_retry_and_never_remove_earlier_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run.jsonl'
            path.write_text('{"id":"a","status":"error"}\n')
            with self.assertRaises(ValueError): recover(path)
            self.assertTrue(recover(path,retry_error=True))
            self.assertEqual(path.read_bytes(),b'')
            raw=b'broken\n{"id":"b","status":"ok"}\n';path.write_bytes(raw)
            with self.assertRaises(ValueError): recover(path,retry_error=True)
            self.assertEqual(path.read_bytes(),raw)

    def test_export_rejects_active_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp)/'raw';raw.mkdir()
            path=raw/'run.jsonl';path.write_text('')
            with run_lock(path):
                with self.assertRaises(RuntimeError): export(raw,Path(tmp)/'bundle',allow_incomplete=True)
            self.assertFalse((Path(tmp)/'bundle').exists())


if __name__=='__main__':unittest.main()
