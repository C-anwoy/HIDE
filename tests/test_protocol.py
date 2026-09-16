import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class ProtocolTests(unittest.TestCase):
    def test_generated_token_likelihood_and_eigen_formula(self):
        import torch
        from hide.baselines import token_statistics, eigen_statistics
        logits = (torch.tensor([[0., 2., -1.]]), torch.tensor([[3., 1., 0.]]))
        ids = torch.tensor([0, 1])  # Deliberately not argmax: catches the original LN-Entropy bug.
        stats = token_statistics(logits, ids)
        expected = -sum(torch.log_softmax(x, -1)[0, t].item() for x, t in zip(logits, ids))/2
        self.assertAlmostEqual(stats['mnll'], expected, places=6)
        z = torch.tensor([[1.,2.,4.],[2.,-1.,3.],[.5,2.,-2.]],dtype=torch.float64)
        result = eigen_statistics(list(z))
        h = torch.eye(3, dtype=z.dtype)-torch.ones(3,3,dtype=z.dtype)/3
        expected = -(torch.linalg.slogdet(z @ h @ z.T + 1e-3*torch.eye(3))[1]/3).item()
        self.assertAlmostEqual(result['negative_eigenscore_paper'], expected, places=5)

    def test_estimator_conventions(self):
        import torch
        from hide.ablations import centered_hsic
        k=torch.randn(4,4); l=torch.randn(4,4)
        self.assertAlmostEqual(float(centered_hsic(k,l,'nminus1')*9), float(centered_hsic(k,l)*16), places=5)
        with self.assertRaises(ValueError): centered_hsic(torch.ones(1,1),torch.ones(1,1),'nminus1')

    def test_suite_commands_parse_and_keep_full_counts(self):
        from hide.launch import command, suite_jobs, catalog
        from hide.runner import parser
        for suite in catalog()[1]['suites']:
            for profile, model, dataset in suite_jobs(suite):
                cmd, output = command(profile, model, dataset)
                args=parser().parse_args(cmd[4:])
                self.assertEqual(args.profile,profile)
                self.assertEqual(args.samples,25 if profile=='pilot' else 200 if profile=='timing' else 0)
                self.assertEqual(args.model_name,model)
                self.assertEqual(output.parent.name,profile)
        self.assertEqual(len(suite_jobs('review')),16)
        self.assertEqual(len(suite_jobs('consistency')),24)
        self.assertEqual(len(suite_jobs('decoding')),24)

    def test_lock_and_atomic_metadata(self):
        from hide.provenance import run_lock, atomic_json
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run.jsonl'
            with run_lock(path):
                with self.assertRaises(RuntimeError):
                    with run_lock(path): pass
            with run_lock(path): pass
            atomic_json(path.with_suffix('.json'), {'ok': True})
            self.assertEqual(json.loads(path.with_suffix('.json').read_text()),{'ok':True})

    def test_seed_is_independent_of_previous_examples(self):
        from hide.provenance import example_seed
        self.assertEqual(example_seed(42,'nq_open','12'), example_seed(42,'nq_open','12'))
        self.assertNotEqual(example_seed(42,'nq_open','12'), example_seed(42,'nq_open','13'))
        self.assertNotEqual(example_seed(42,'nq_open','12',0), example_seed(42,'nq_open','12',1))


if __name__=='__main__': unittest.main()
