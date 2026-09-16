import json
import hashlib
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from hide.execution import RunControl, RunPaused
from hide.export_results import export, inspect_run, restore
from hide.launch import profile_arguments
from hide.parts import (choose_tasks, execute_child, load_plan, make_plan, merge,
                        part_path, portable_sources)
from hide.provenance import atomic_json, source_files


class PartTests(unittest.TestCase):
    def fixture(self, root):
        plan = make_plan('review', 2)
        plan_path = root/'plan.json'; atomic_json(plan_path,plan)
        queue=root/'queue'
        for task in plan['tasks']:
            path=part_path(queue,task); path.parent.mkdir(parents=True,exist_ok=True)
            parent=['0','1','2','3','4']
            selected=parent[task['start']:task['stop']]
            args=profile_arguments('qa')
            args.update(profile='qa',model_name='llama3-8b',dataset='nq_open',layer=16,
                        attention_backend='eager',dtype='bfloat16',start=task['start'],stop=task['stop'],
                        output=str(path), model_path='/weights/model',data_root='/datasets',
                        keyword_model='/weights/kw',judge_model='/weights/judge',device='cuda:0')
            meta=dict(arguments=args,selected_ids=selected,source_sha256=source_files(),
                      cohort_sha256='part'+str(task['start']),schema_version=2,gpu='A100',
                      packages={'torch':'2.5.1'},checkpoint_config={},checkpoint_identity={},
                      partition=dict(start=task['start'],stop=task['stop'],parent_selected_ids=parent,
                                     parent_cohort_sha256=hashlib.sha256(b''.join(json.dumps([i,'question '+i,'answer '+i],ensure_ascii=False).encode() for i in parent)).hexdigest()))
            atomic_json(path.with_suffix('.manifest.json'),meta)
            atomic_json(path.with_suffix('.runtime.json'),dict(hostname='host',gpu_uuid='uuid'))
            snapshots=path.with_suffix('.sources'); snapshots.mkdir()
            for source in meta['source_sha256']:
                src=Path(source); (snapshots/(src.parent.name+'__'+src.name)).write_bytes(src.read_bytes())
            path.write_text(''.join(json.dumps(dict(id=i,status='ok',HIDE_score=float(i),
                                                   prompt='question '+i,answer='answer '+i,
                                                   example_wall_s=.1))+'\n' for i in reversed(selected)))
            path.with_suffix('.log').write_text('Raw run log\n')
        return plan_path,plan,queue

    def small_plan(self):
        return patch('hide.parts.suite_jobs',return_value=[('qa','llama3-8b','nq_open')])

    def test_plan_full_coverage_no_redundant_qa_and_static_assignments(self):
        plan=make_plan('full')
        tasks=plan['tasks']
        self.assertFalse(any(t['profile']=='qa' for t in tasks))
        self.assertEqual(len([t for t in tasks if t['profile']=='timing']),8)
        from collections import defaultdict
        groups=defaultdict(list)
        for t in tasks:
            groups[(t['profile'],t['model'],t['dataset'])].append(t)
            self.assertLessEqual(t['stop']-t['start'],200)
        from hide.export_results import DATASET_COUNTS
        for (profile,_,ds),parts in groups.items():
            self.assertEqual(parts[0]['start'],0)
            self.assertEqual(parts[-1]['stop'],200 if profile=='timing' else DATASET_COUNTS[ds])
            self.assertTrue(all(a['stop']==b['start'] for a,b in zip(parts,parts[1:])))
        shards=[{t['id'] for t in choose_tasks(plan,index=i,workers=3)} for i in range(3)]
        self.assertEqual(len(set.union(*shards)),sum(map(len,shards)))
        self.assertEqual(set.union(*shards),{t['id'] for t in tasks if t['profile']!='timing'})

    def test_merge_exact_order_provenance_export_and_restore(self):
        with tempfile.TemporaryDirectory() as tmp,self.small_plan(),patch.dict('hide.parts.DATASET_COUNTS',{'nq_open':5}):
            root=Path(tmp); path,plan,queue=self.fixture(root)
            # Device names can differ for accuracy; each original manifest is retained.
            part=part_path(queue,plan['tasks'][1]).with_suffix('.manifest.json')
            meta=json.loads(part.read_text()); meta['gpu']='different GPU'; atomic_json(part,meta)
            output=root/'merged'; merge(path,[queue],output)
            merged=output/'qa'/'llama3-8b_nq_open.jsonl'
            self.assertTrue(inspect_run(merged)['complete'])
            self.assertEqual([json.loads(line)['id'] for line in merged.read_text().splitlines()],list('01234'))
            self.assertEqual(len(list(output.rglob('*.jsonl'))),1)
            for task in plan['tasks']:
                raw=part_path(queue,task)
                saved=output/'provenance'/'parts'/task['id']/'qa'/raw.with_suffix('.jsonl.bak').name
                self.assertEqual(saved.read_bytes(),raw.read_bytes())
            export(output,root/'bundle')
            restore(root/'bundle',root/'restored')
            self.assertEqual(merged.read_bytes(),(root/'restored'/'qa'/merged.name).read_bytes())

    def test_merge_rejects_missing_duplicate_and_incompatible_parts(self):
        with tempfile.TemporaryDirectory() as tmp,self.small_plan(),patch.dict('hide.parts.DATASET_COUNTS',{'nq_open':5}):
            root=Path(tmp); path,plan,queue=self.fixture(root)
            part=part_path(queue,plan['tasks'][0]); raw=part.read_bytes()
            part.unlink()
            with self.assertRaisesRegex(ValueError,'exactly one'): merge(path,[queue],root/'missing')
            part.write_bytes(raw)
            import shutil
            other=root/'other'; shutil.copytree(queue,other)
            with self.assertRaisesRegex(ValueError,'exactly one'): merge(path,[queue,other],root/'duplicate')
            meta_path=part.with_suffix('.manifest.json'); meta=json.loads(meta_path.read_text())
            meta['packages']['torch']='changed'; atomic_json(meta_path,meta)
            with self.assertRaisesRegex(ValueError,'Incompatible'): merge(path,[queue],root/'incompatible')
            self.assertFalse((root/'incompatible').exists())

    def test_shared_queue_lock_excludes_exports_and_merge(self):
        from hide.provenance import queue_lock
        with tempfile.TemporaryDirectory() as tmp:
            with queue_lock(tmp,shared=True):
                with queue_lock(tmp,shared=True): pass
                with self.assertRaises(RuntimeError):
                    with queue_lock(tmp): pass
            with queue_lock(tmp):
                with self.assertRaises(RuntimeError):
                    with queue_lock(tmp,shared=True): pass

    def test_worker_pauses_then_resumes_without_duplicate_rows(self):
        import argparse
        from hide.parts import work
        with tempfile.TemporaryDirectory() as tmp,self.small_plan(),patch.dict('hide.parts.DATASET_COUNTS',{'nq_open':5}):
            root=Path(tmp); plan_path,plan,queue=self.fixture(root)
            first=part_path(queue,plan['tasks'][0]); full=first.read_bytes()
            first.write_bytes(full.splitlines(keepends=True)[0])
            cmd=[sys.executable,'--model-path',str(root),'--keyword-model',str(root),'--judge-model',str(root)]
            args=argparse.Namespace(plan=str(plan_path),queue=str(queue),kind='detection',worker_index=0,
                                   workers=1,hours=.01,hard_hours=.02,task=None,max_parts=1,models=None,profiles=None)
            with patch('hide.parts.command',return_value=(cmd,first)),patch('hide.parts.execute_child',return_value=75) as child:
                work(args)
                self.assertEqual(child.call_count,1)
            self.assertFalse(inspect_run(first)['complete'])
            self.assertFalse(first.with_suffix('.failed.json').exists())
            def finish(*_):
                first.write_bytes(full)
                return 0
            with patch('hide.parts.command',return_value=(cmd,first)),patch('hide.parts.execute_child',side_effect=finish) as child:
                work(args)
                self.assertEqual(child.call_count,1)
            self.assertTrue(inspect_run(first)['complete'])
            with patch('hide.parts.execute_child') as child:
                work(args)
                child.assert_not_called()

    def test_plan_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plan.json'; plan=make_plan('review'); atomic_json(path,plan)
            plan['tasks'][0]['stop']+=1; atomic_json(path,plan)
            with self.assertRaisesRegex(ValueError,'modified'): load_plan(path)

    def test_timing_device_requires_exclusive_fixed_hardware(self):
        from hide.parts import timing_device
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-test','HIDE_DEVICE':'cuda:0'}):
            plan={'sha256':'plan'}
            with patch('hide.parts.subprocess.check_output',side_effect=['GPU-test, A100, 555, 150\n','']):
                timing_device(tmp,plan)
            with patch('hide.parts.subprocess.check_output',side_effect=['GPU-test, A100, 555, 150\n','GPU-test, 123\n']):
                with self.assertRaisesRegex(ValueError,'active compute'): timing_device(tmp,plan)
            with patch('hide.parts.subprocess.check_output',side_effect=['GPU-test, A100, 555, 250\n','']):
                with self.assertRaisesRegex(ValueError,'power limit changed'): timing_device(tmp,plan)

    def test_queue_snapshot_detects_missing_plan_tasks(self):
        with tempfile.TemporaryDirectory() as tmp,self.small_plan(),patch.dict('hide.parts.DATASET_COUNTS',{'nq_open':5}):
            root=Path(tmp); plan_path,plan,queue=self.fixture(root)
            atomic_json(queue/'plan.json',plan)
            part_path(queue,plan['tasks'][0]).unlink()
            with self.assertRaisesRegex(ValueError,'unfinished/missing parts'):
                export(queue,root/'bundle')
            index=export(queue,root/'checkpoint',allow_incomplete=True)
            self.assertFalse(index['complete'])
            objects=[part['path'] for entry in index['files'] for part in entry['parts'] if 'source_objects' in part['path']]
            self.assertGreater(len(objects),len(set(objects)))

    def test_control_pause_and_signal_restoration(self):
        previous=signal.getsignal(signal.SIGTERM)
        with RunControl() as control:
            control.check(); control.request()
            with self.assertRaises(RunPaused): control.check()
        self.assertIs(signal.getsignal(signal.SIGTERM),previous)
        with patch('hide.execution.time.monotonic',return_value=1): control=RunControl(1)
        with patch('hide.execution.time.monotonic',return_value=3):
            with self.assertRaises(RunPaused): control.check()

    def test_worker_supervisor_hard_and_cooperative_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); events=[]; now=time.monotonic()
            code=execute_child([sys.executable,'-c',
                'import signal,time,sys; signal.signal(signal.SIGTERM,lambda *_: sys.exit(75)); print("ready",flush=True); time.sleep(30)'],
                root/'soft.log',now+.5,now+3,events)
            self.assertEqual(code,75)
            events=[]; now=time.monotonic()
            code=execute_child([sys.executable,'-c',
                'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)'],
                root/'hard.log',now+.5,now+1,events)
            self.assertEqual(code,-signal.SIGKILL)
            self.assertTrue(any(e['event']=='hard_stop' for e in events))


if __name__=='__main__': unittest.main()
