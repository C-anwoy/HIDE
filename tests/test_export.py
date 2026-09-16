import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from hide.export_results import export, verify, restore, suite_issues


class ExportTests(unittest.TestCase):
    def make_run(self, root):
        path=root/'detection'/'tiny.jsonl'
        path.parent.mkdir(parents=True)
        manifest={'arguments':{'mode':'detection','model_name':'tiny','dataset':'nq_open','samples':0,'repeats':3},
                  'selected_ids':['a','b'], 'source_sha256':{}}
        path.with_suffix('.manifest.json').write_text(json.dumps(manifest))
        rows=[dict(id=i,status='ok',example_wall_s=.3,HIDE_score=.2) for i in ['a','b']]
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return path

    def test_lossless_shards_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); raw=root/'raw'; path=self.make_run(raw)
            (raw/'notes.md').write_text('All examples are retained.\n')
            with patch('hide.export_results.CHUNK_BYTES',50):
                index=export(raw,root/'bundle')
            self.assertTrue(index['complete'])
            self.assertGreater(len(next(e for e in index['files'] if e['source'].endswith('.jsonl'))['parts']),1)
            restore(root/'bundle',root/'restored')
            for original in raw.rglob('*'):
                if original.is_file():
                    self.assertEqual(original.read_bytes(),(root/'restored'/original.relative_to(raw)).read_bytes())
            entry=index['files'][0]['parts'][0]
            part=root/'bundle'/entry['path']; part.write_bytes(part.read_bytes()+b'changed')
            with self.assertRaises(ValueError): verify(root/'bundle')

    def test_incomplete_export_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); raw=root/'raw'; path=self.make_run(raw)
            path.write_text(json.dumps({'id':'a','status':'ok'})+'\n')
            with self.assertRaises(ValueError): export(raw,root/'bundle')
            self.assertFalse((root/'bundle').exists())
            index=export(raw,root/'checkpoint',allow_incomplete=True)
            self.assertFalse(index['complete'])
            self.assertEqual(index['runs'][0]['missing_keys'],[('b',0)])
            self.assertTrue(suite_issues(index['runs']))

    def test_required_full_suite(self):
        from hide.export_results import DATASET_COUNTS
        from hide.launch import suite_jobs, profile_arguments, catalog
        models, _ = catalog()
        runs=[]
        for profile, model, ds in suite_jobs('review'):
            args = profile_arguments(profile)
            args.update(layer=models[model]['layer'], attention_backend='eager', dtype='bfloat16')
            n = args['samples'] or DATASET_COUNTS[ds]
            count = n * (args['repeats'] if args['mode']=='timing' else 1)
            runs.append(dict(mode=args['mode'], model=model, dataset=ds, arguments=args,
                             expected=count, complete=True))
        self.assertFalse(suite_issues(runs))
        runs.pop()
        self.assertEqual(len(suite_issues(runs)),1)
        runs[0]['arguments']['keywords']=10
        self.assertEqual(len(suite_issues(runs)),2)


if __name__=='__main__': unittest.main()
