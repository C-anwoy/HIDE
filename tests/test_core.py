import ast
import json
from pathlib import Path
import types
import unittest

import numpy as np
import pandas as pd
from hide.evaluate import auc, pcc, evaluate, normalize_text
from hide.summarize_timing import summarize


class EvaluationTests(unittest.TestCase):
    def test_auc_matches_sklearn_with_ties(self):
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(10)
        for _ in range(20):
            y = rng.integers(0, 2, 100)
            scores = rng.integers(0, 6, 100)
            self.assertAlmostEqual(auc(y, scores), roc_auc_score(y, scores))
        self.assertEqual(auc(np.array([0,1]), np.ones(2)), .5)
        self.assertTrue(np.isnan(auc(np.zeros(4), np.arange(4))))

    def test_paired_bootstrap_and_score_filtering(self):
        df = pd.DataFrame(dict(id=list('abcdefgh'), HIDE_score=[0,.0001,.2,.3,.4,.5,.6,.7],
                              Omega=[0,.0001,.2,.3,.4,.5,.6,.7], Delta_in=np.arange(8),
                              sentence_similarity=np.arange(8)/7, is_correct=[0]*7+[1]))
        result, diagnostics = evaluate(df,bootstrap=100)
        same = result[result.method == 'Omega'].iloc[0]
        self.assertEqual(same['n'],8)
        self.assertEqual(same.hide_minus_baseline_auc_lo,0)
        self.assertEqual(same.hide_minus_baseline_auc_hi,0)
        self.assertIn('negative_Delta_in',result.method.to_list())
        self.assertEqual(diagnostics[0]['excluded'],0)
        df.loc[0,'Omega'] = np.nan
        result, diagnostics = evaluate(df,bootstrap=0)
        self.assertTrue((result.n == 7).all())
        self.assertEqual(diagnostics[0]['excluded_ids'],['a'])
        df.loc[1,'id'] = 'a'
        with self.assertRaises(ValueError):
            evaluate(df,bootstrap=0)

    def test_normalization(self):
        self.assertEqual(normalize_text(' The,   Eiffel Tower!'), 'eiffel tower')

    def test_timing_accounting_and_negative_overhead(self):
        df = pd.DataFrame(dict(id=['a','a','b','b'],repeat=[0,1,0,1],
            base_s=[2,2,2,2],capture_s=[2,2,1.8,1.8],score_s=[.1]*4,
            total_s=[2.1,2.1,1.9,1.9],overhead_s=[.1,.1,-.1,-.1],status=['ok']*4))
        row, errors = summarize(df,bootstrap=100)
        self.assertEqual(row['n_queries'],2)
        self.assertEqual(row['n_measurements'],4)
        self.assertAlmostEqual(row['overhead_pct'],0)
        self.assertFalse(errors)


class CoreTests(unittest.TestCase):
    def test_symbol_answer_uses_existing_token_fallback(self):
        import torch
        from hide.core import get_unbiased_hsic_score_keybert
        class Tok:
            def decode(self, ids, **kwargs):
                return ''.join({1: 'question', 649: ' *', 9: '\n'}[int(i)] for i in ids)
            def encode(self, text, **kwargs):
                return [1] if text.strip() == 'question' else [649]
        class KW:
            def extract_keywords(self, text, **kwargs):
                # Actual sklearn empty-vocabulary failure used by KeyBERT's custom vectorizer.
                vectorizer = kwargs['vectorizer'].fit([text])
                return [(word, 1.0) for word in vectorizer.get_feature_names_out()]
        hidden = [tuple(torch.randn(1, 1, 8) for _ in range(3)) for _ in range(2)]
        score, ik, ok, it, ot = get_unbiased_hsic_score_keybert(
            hidden, Tok(), torch.tensor([1]), torch.tensor([649, 9]), 20, 1, kw_model=KW())
        self.assertEqual(ok, [])
        self.assertEqual(ot, [' *'])
        self.assertEqual(len(it), 1)
        self.assertEqual(score, 0.0)  # The unchanged estimator yields zero for n_eff=1.

    def test_empty_vocabulary_guard_does_not_hide_other_errors(self):
        import torch
        from hide.core import extract_keyword_representation
        class Tok:
            def decode(self, ids, **kwargs):
                return 'word'
        class KW:
            def __init__(self, error):
                self.error = error
            def extract_keywords(self, text, **kwargs):
                raise self.error
        for error in (ValueError('empty vocabulary; unexpected on ordinary words'),
                      RuntimeError('out of memory'), ValueError('invalid embeddings')):
            with self.assertRaises(RuntimeError) as caught:
                extract_keyword_representation(torch.ones(1, 2), torch.ones(1, 2), Tok(),
                                               torch.tensor([1]), torch.tensor([1]), kw_model=KW(error))
            self.assertIs(caught.exception.__cause__, error)
            self.assertIn(str(error), str(caught.exception))

    def test_formula_preserved_and_constant_kernel_control(self):
        import torch
        from hide.core import unbiased_HSIC
        source = ast.parse(Path('archive/original/func/metric.py').read_text())
        fn = next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='unbiased_HSIC')
        scope = {'torch':torch}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'original_metric','exec'),scope)
        for n in [1,2,3,4,20]:
            k = torch.ones(n,n)
            self.assertAlmostEqual(unbiased_HSIC(k,k).item(),(n-1)/n**2,places=6)
            x,y = torch.randn(n,n),torch.randn(n,n)
            torch.testing.assert_close(unbiased_HSIC(x,y),scope['unbiased_HSIC'](x,y),rtol=0,atol=0)

    def test_token_selection_and_full_score_parity(self):
        import torch
        from sklearn.feature_extraction.text import CountVectorizer
        from hide.kernels import KERNEL_FUNCTIONS
        from hide.core import get_unbiased_hsic_score_keybert
        class Tok:
            def decode(self, ids, **kw):
                return ' '.join(str(int(x)) for x in ids)
            def encode(self, word, **kw):
                return [int(word.strip())]
        class KW:
            def extract_keywords(self,text,**kw):
                return [(w,1.) for w in dict.fromkeys(text.split())][:kw['top_n']]
        original = ast.parse(Path('archive/original/func/metric.py').read_text())
        names=['extract_keyword_representation','unbiased_HSIC','compute_unbiased_hsic_with_keywords',
               'get_unbiased_hsic_score_keybert']
        body=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in names]
        scope=dict(torch=torch, CountVectorizer=CountVectorizer, KERNEL_FUNCTIONS=KERNEL_FUNCTIONS,
                   KeyBERT=lambda **kw: KW(), os=__import__('os'), _settings=types.SimpleNamespace(MODEL_PATH='/tmp'))
        exec(compile(ast.Module(body=body,type_ignores=[]),'original_metric','exec'),scope)
        hidden=[tuple(torch.randn(1,5 if step==0 else 1,8) for _ in range(4)) for step in range(4)]
        params=dict(hidden_states=hidden,tokenizer=Tok(),input_tokens=torch.tensor([1,2,1,3,4]),
                    output_tokens=torch.tensor([2,3,2,9]),keywords=6,layer=2)
        a = scope['get_unbiased_hsic_score_keybert'](**params)
        b = get_unbiased_hsic_score_keybert(**params,kw_model=KW())
        self.assertEqual(a,b)  # Includes duplicated token positions and ordering.
        params['hidden_states']=hidden[:1]
        params['output_tokens']=torch.tensor([9])
        self.assertEqual(get_unbiased_hsic_score_keybert(**params,kw_model=KW())[0],0)

    def test_cached_alignment_llama_and_gemma(self):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM,Gemma2Config,Gemma2ForCausalLM, StoppingCriteria
        from hide.runner import proxies
        for cls, cfg in [(LlamaForCausalLM,LlamaConfig),(Gemma2ForCausalLM,Gemma2Config)]:
            config=cfg(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
                       num_attention_heads=4,num_key_value_heads=2,head_dim=8,
                       max_position_embeddings=64,pad_token_id=0,eos_token_id=None,
                       sliding_window=16,attn_implementation='eager')
            model=cls(config).eval()
            ids=torch.tensor([[2,3,4,5]])
            with torch.no_grad():
                class StopAtEight(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        return input_ids.shape[1] >= 8
                # Stop before the allocated cache boundary: transformers 4.51.3 HybridCache
                # shifts its sliding cache one step early at the boundary (see audit).
                result=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=6,
                    stopping_criteria=[StopAtEight()],do_sample=False,return_dict_in_generate=True,
                    output_hidden_states=True,output_attentions=True)
                replay=model(result.sequences,attention_mask=torch.ones_like(result.sequences),
                             output_hidden_states=True,output_attentions=True,use_cache=False)
            generated=torch.cat([s[2][0] for s in result.hidden_states[1:]])
            torch.testing.assert_close(generated,replay.hidden_states[2][0,4:-1],atol=1e-5,rtol=1e-4)
            omega,delta=proxies(result.hidden_states,result.attentions,4,2)
            attention=replay.attentions[1][0,:,4:-1,:4].float().mean(0)
            self.assertAlmostEqual(omega,attention.sum(-1).mean().item(),places=5)
            self.assertAlmostEqual(delta,(attention @ replay.hidden_states[1][0,:4].float()).norm(dim=-1).mean().item(),places=5)


if __name__=='__main__':
    unittest.main()
