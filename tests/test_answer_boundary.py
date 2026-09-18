import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hide.answer_boundary import first_answer_line, generation_fields, stopping_criteria


class Tokenizer:
    pieces = {0:'', 1:'Question\nA:', 2:'Paris', 3:'\n\n', 4:'Q:', 5:'\r\nNext', 6:'\n'}
    def decode(self, ids, **kwargs):
        return ''.join(self.pieces[int(i)] for i in ids)


class AnswerBoundaryTests(unittest.TestCase):
    def test_first_nonempty_line_and_terminators(self):
        self.assertEqual(first_answer_line(' \n\n Paris \r\nQ: next'), ('Paris',True,'Q: next'))
        self.assertEqual(first_answer_line(' *'), ('*',False,''))
        self.assertEqual(first_answer_line('\n \n'), ('',False,''))
        self.assertEqual(first_answer_line(' Paris\n'), ('Paris',True,''))
        self.assertEqual(generation_fields(Tokenizer(), [2,3])['evaluated_text'], 'Paris')
        self.assertEqual(generation_fields(Tokenizer(), [2,5])['boundary_token_suffix'], 'Next')
        with self.assertRaisesRegex(ValueError,'before the final token'):
            generation_fields(Tokenizer(), [2,3,4])

    def test_prompt_newlines_ignored_and_double_newline_token_detected(self):
        import torch
        criteria = stopping_criteria(Tokenizer(), 1)
        self.assertFalse(criteria(torch.tensor([[1,2]]),None).item())
        self.assertTrue(criteria(torch.tensor([[1,2,3]]),None).item())
        self.assertFalse(criteria(torch.tensor([[1,6]]),None).item())
        self.assertTrue(criteria(torch.tensor([[1,6,2,3]]),None).item())

    def test_real_llama_gemma_cached_generation_stops_before_next_question(self):
        import torch
        from transformers import (LlamaConfig,LlamaForCausalLM,Gemma2Config,Gemma2ForCausalLM,
                                  LogitsProcessor,LogitsProcessorList)
        class ForcedAnswer(LogitsProcessor):
            def __call__(self, input_ids, scores):
                scores.fill_(-float('inf'))
                scores[:,[2,3,4][min(input_ids.shape[1]-1,2)]] = 0
                return scores
        for cls, model_cls in [(LlamaConfig,LlamaForCausalLM),(Gemma2Config,Gemma2ForCausalLM)]:
            cfg=cls(vocab_size=16,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
                    num_attention_heads=4,num_key_value_heads=2,head_dim=8,sliding_window=32,
                    eos_token_id=None,pad_token_id=0)
            cfg._attn_implementation='eager'
            model=model_cls(cfg).eval()
            with torch.inference_mode():
                out=model.generate(torch.tensor([[1]]),attention_mask=torch.ones(1,1,dtype=torch.long),
                    max_new_tokens=5,do_sample=False,use_cache=True,return_dict_in_generate=True,
                    output_hidden_states=True,output_attentions=True,
                    logits_processor=LogitsProcessorList([ForcedAnswer()]),
                    stopping_criteria=stopping_criteria(Tokenizer(),1))
            self.assertEqual(out.sequences.tolist(),[[1,2,3]])
            self.assertEqual(len(out.hidden_states),2)
            self.assertEqual(len(out.attentions),2)
            self.assertEqual(generation_fields(Tokenizer(),out.sequences[0,1:])['evaluated_text'],'Paris')

    def test_answer_plan_covers_full_cohorts_and_timing_is_intact(self):
        from hide.parts import make_plan,choose_tasks,is_timing
        p=make_plan('answer-review')
        self.assertEqual(len(p['tasks']),242)
        detection=choose_tasks(p,'detection'); timing=choose_tasks(p,'timing')
        self.assertEqual(len(detection),234)
        self.assertEqual(sum(t['stop']-t['start'] for t in detection),45992)
        self.assertEqual(len(timing),8)
        self.assertTrue(all(is_timing(t['profile']) and t['start']==0 and t['stop']==200 for t in timing))
        self.assertEqual(len(make_plan('answer-pilots')['tasks']),8)

    def test_pilot_gate_preserves_low_accuracy_but_blocks_wrong_boundary_metadata(self):
        from hide.answer_runs import initialize,check_pilots
        from hide.parts import load_plan,part_path
        with tempfile.TemporaryDirectory() as td:
            root=initialize(td);plan=load_plan(root/'pilot/plan.json')
            for task in plan['tasks']:
                path=part_path(root/'pilot',task);path.parent.mkdir(parents=True,exist_ok=True)
                row=dict(id='test',generated_text='Wrong answer\n',evaluated_text='Wrong answer',
                         answer_boundary='first-line',answer_boundary_reached=True,boundary_token_suffix='',
                         is_correct=0,HIDE_score=.1,Omega=.1,Delta_in=.1,sentence_similarity=0.,rouge_l=0.)
                path.write_text(json.dumps(row)+'\n')
                from hide.provenance import source_files
                path.with_suffix('.manifest.json').write_text(json.dumps(dict(
                    arguments={'answer_boundary':'first-line'},source_sha256=source_files())))
            with patch('hide.answer_runs.inspect_run',return_value={'complete':True}),contextlib.redirect_stdout(io.StringIO()):
                results=check_pilots(root,['llama3-8b'])
                self.assertEqual(len(results),4)
                self.assertTrue(all(r['n_correct']==0 for r in results))
                path=part_path(root/'pilot',plan['tasks'][0]);row=json.loads(path.read_text())
                row['evaluated_text']='silently altered';path.write_text(json.dumps(row)+'\n')
                with self.assertRaisesRegex(ValueError,'Pilot gate stopped'):
                    check_pilots(root,['llama3-8b'])

    def test_pilot_and_detection_share_one_deadline_and_keep_dataset_filter(self):
        from hide.answer_runs import bounded_work
        clock=[0.]; calls=[]
        def work(args):
            calls.append(args);clock[0]+=3600
        with patch('hide.answer_runs.time.monotonic',side_effect=lambda:clock[0]), \
             patch('hide.answer_runs.work',side_effect=work):
            self.assertTrue(bounded_work('/tmp/test',['llama3-8b'],True,0,2,4))
            self.assertFalse(bounded_work('/tmp/test',['llama3-8b'],False,0,2,4,datasets=['nq_open']))
        self.assertEqual([(x.hours,x.hard_hours) for x in calls],[(2,4),(1,3)])
        self.assertIsNone(calls[0].datasets)
        self.assertEqual(calls[1].datasets,['nq_open'])

    def test_timing_waiter_waits_for_all_parts_and_stops_on_failure(self):
        from hide.answer_runs import wait_detection
        from hide.parts import part_path
        tasks=[dict(id=f'test-{i}',profile='answer-qa',model='llama3-8b',dataset=ds)
               for i,ds in enumerate(['nq_open','SQuAD'])]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);clock=[0.];calls=[]
            for t in tasks:
                p=part_path(root,t);p.parent.mkdir(parents=True,exist_ok=True);p.touch()
            def inspect(path):
                calls.append(path.name)
                return {'complete': 'nq_open' in path.name or clock[0]>=15}
            with patch('hide.answer_runs.load_plan',return_value={'tasks':tasks}), \
                 patch('hide.answer_runs.inspect_run',side_effect=inspect), \
                 patch('hide.answer_runs.time.monotonic',side_effect=lambda:clock[0]), \
                 patch('hide.answer_runs.time.sleep',side_effect=lambda s:clock.__setitem__(0,clock[0]+s)), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(wait_detection(root,100))
                self.assertEqual(calls.count('llama3-8b_nq_open.jsonl'),1)
                self.assertEqual(calls.count('llama3-8b_SQuAD.jsonl'),2)
                part_path(root,tasks[0]).with_suffix('.failed.json').write_text('{}')
                with self.assertRaisesRegex(RuntimeError,'Detection failed'):
                    wait_detection(root,100)


if __name__ == '__main__':
    unittest.main()
