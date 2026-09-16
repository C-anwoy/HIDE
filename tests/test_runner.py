"""Offline end-to-end wiring test. Fake data/keyword/judge services; real tiny HF generation."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class RunnerTests(unittest.TestCase):
    def test_detection_timing_and_resume(self):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        from hide.runner import main
        cfg=LlamaConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
                        num_attention_heads=4,num_key_value_heads=2,eos_token_id=None,pad_token_id=0)
        cfg._attn_implementation='eager'
        torch.manual_seed(5)
        model=LlamaForCausalLM(cfg).eval()
        class Tokenizer:
            eos_token_id=0
            pad_token_id=0
            def decode(self, ids, **kwargs):
                return ' '.join(str(int(x)) for x in ids)
            def encode(self, text, **kwargs):
                return [int(text.strip())]
        class Data:
            def __init__(self, rows=None):
                self.rows=[dict(id=str(i),input_ids=torch.tensor([2,3,4,5]),attention_mask=torch.ones(4,dtype=torch.long),
                    answer='7',question='2 3',prompt='2 3 4 5') for i in range(2)] if rows is None else rows
            def shuffle(self, **kwargs): return self
            def select(self, indices): return Data([self.rows[i] for i in indices])
            def __len__(self): return len(self.rows)
            def __iter__(self): return iter(self.rows)
            def __getitem__(self,key):
                return [r[key] for r in self.rows] if isinstance(key,str) else self.rows[key]
        ds=types.ModuleType('hide.datasets.nq_open')
        ds.__file__=__file__
        ds.get_dataset=lambda tokenizer: Data()
        ds._generate_config=lambda tokenizer: dict(eos_token_id=None)
        class Embeddings:
            def __init__(self,*args,**kwargs): pass
            def encode(self,texts,**kwargs):
                return torch.tensor([[float(len(s)+1),1.] for s in texts])
        class KW:
            def __init__(self,**kwargs): pass
            def extract_keywords(self,text,**kw):
                return [(w,1.) for w in dict.fromkeys(text.split())][:kw['top_n']]
        class Rouge:
            def __init__(self,*args,**kwargs): pass
            def score(self,**kwargs): return {'rougeL':types.SimpleNamespace(fmeasure=.1)}
        def cos_sim(a,b):
            a=a.reshape(-1,a.shape[-1]); b=b.reshape(-1,b.shape[-1])
            return torch.nn.functional.normalize(a) @ torch.nn.functional.normalize(b).T
        modules={'hide.datasets.nq_open':ds,
                 'sentence_transformers':types.SimpleNamespace(SentenceTransformer=Embeddings,util=types.SimpleNamespace(cos_sim=cos_sim)),
                 'keybert':types.SimpleNamespace(KeyBERT=KW),
                 'rouge_score':types.ModuleType('rouge_score'),
                 'rouge_score.rouge_scorer':types.SimpleNamespace(RougeScorer=Rouge)}
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules,modules), \
             patch.dict(os.environ,{},clear=False), \
             patch.dict('hide.export_results.DATASET_COUNTS', {'nq_open': 2}), \
             patch('transformers.AutoConfig.from_pretrained',return_value=cfg), \
             patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()), \
             patch('transformers.AutoModelForCausalLM.from_pretrained',return_value=model):
            for label, mode, extra in [('detection','detection',[]), ('timing','timing',[]),
                                        ('ablations','detection',['--ablations']),
                                        ('comparison','detection',['--multipass-samples','3']),
                                        ('nucleus','detection',['--decoding','nucleus','--top-p','.8'])]:
                output=Path(root)/(label+'.jsonl')
                weights=Path(root)/'weights'; weights.mkdir(exist_ok=True)
                argv=['runner','--mode',mode,'--model-path',str(weights),'--model-name','tiny-test',
                      '--dataset','nq_open','--data-root',root,'--keyword-model',str(weights),'--judge-model',str(weights),
                      '--device','cpu','--dtype','float32','--samples','2','--layer','2',
                      '--max-new-tokens','4','--warmup','1','--repeats','2','--output',str(output)] + extra
                with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                    main()
                before=output.read_text()
                rows=[json.loads(x) for x in before.splitlines()]
                self.assertEqual(len(rows),2 if mode=='detection' else 4)
                self.assertTrue(all(r['status']=='ok' for r in rows))
                with patch.object(sys,'argv',argv+['--resume']),contextlib.redirect_stdout(io.StringIO()):
                    main()
                self.assertEqual(output.read_text(),before)
                # A partition must preserve generation and score for the same seeded example.
                if label != 'timing':
                    partition_argv = argv.copy()
                    partition_output = Path(root)/(label+'_part.jsonl')
                    partition_argv[partition_argv.index('--output')+1] = str(partition_output)
                    with patch.object(sys,'argv',partition_argv+['--start','1','--stop','2']), contextlib.redirect_stdout(io.StringIO()):
                        main()
                    partition_rows = [json.loads(x) for x in partition_output.read_text().splitlines()]
                    self.assertEqual(len(partition_rows),1)
                    self.assertEqual(partition_rows[0]['id'], rows[1]['id'])
                    self.assertEqual(partition_rows[0]['generated_ids'], rows[1]['generated_ids'])
                    self.assertEqual(partition_rows[0]['HIDE_score'], rows[1]['HIDE_score'])
                    if label == 'comparison':
                        self.assertEqual(partition_rows[0]['multipass_samples'],rows[1]['multipass_samples'])
                if label == 'nucleus':
                    # Cooperative time stop writes no error row and resumes the same stochastic cohort.
                    from hide.execution import RunPaused
                    paused_output=Path(root)/'paused.jsonl'
                    paused_argv=argv.copy(); paused_argv[paused_argv.index('--output')+1]=str(paused_output)
                    checks=[0]
                    def budget_check(_):
                        checks[0]+=1
                        if checks[0]==4: raise RunPaused('test budget')
                    with patch.object(sys,'argv',paused_argv), patch('hide.execution.RunControl.check',budget_check), contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(SystemExit) as paused: main()
                    self.assertEqual(paused.exception.code,75)
                    paused_rows=[json.loads(x) for x in paused_output.read_text().splitlines()]
                    self.assertEqual(len(paused_rows),1)
                    self.assertEqual(paused_rows[0]['status'],'ok')
                    with patch.object(sys,'argv',paused_argv+['--resume','--time-budget-seconds','10000']), contextlib.redirect_stdout(io.StringIO()):
                        main()
                    resumed=[json.loads(x) for x in paused_output.read_text().splitlines()]
                    self.assertEqual([r['generated_ids'] for r in resumed],[r['generated_ids'] for r in rows])
                if label == 'ablations':
                    self.assertTrue(all(r['ablations'] for r in rows))
                    self.assertTrue(all(v['status'] != 'error' for r in rows for v in r['ablations']))
                if label == 'comparison':
                    self.assertTrue(all(len(r['multipass_samples']) == 3 for r in rows))
                if label == 'nucleus':
                    # Recreate an interrupted file, then verify that later stochastic records are identical.
                    output.write_text(json.dumps(rows[0])+'\n')
                    with patch.object(sys,'argv',argv+['--resume']), contextlib.redirect_stdout(io.StringIO()):
                        main()
                    regenerated = [json.loads(x) for x in output.read_text().splitlines()]
                    self.assertEqual(regenerated[1]['generated_ids'], rows[1]['generated_ids'])
                    self.assertEqual(regenerated[1]['HIDE_score'], rows[1]['HIDE_score'])
                if mode=='timing':
                    self.assertTrue(all(abs(r['total_s']-r['base_s']-r['overhead_s'])<1e-10 for r in rows))


if __name__=='__main__': unittest.main()
