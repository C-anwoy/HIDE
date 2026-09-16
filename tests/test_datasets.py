"""Offline fixture tests for prompt, stop and first-reference contracts."""
import ast
import copy
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class DatasetTests(unittest.TestCase):
    def test_original_prompt_and_stop_parity(self):
        class Tokenizer:
            eos_token_id=99
            def __call__(self, text, add_special_tokens=True):
                return {'input_ids':([98] if add_special_tokens else [])+[ord(c) for c in text]}
            def encode(self, text): return self(text)['input_ids']
        fake=types.SimpleNamespace(Dataset=object)
        with patch.dict(sys.modules,{'datasets':fake}):
            for name in ['SQuAD','race','nq_open','triviaqa']:
                module=importlib.import_module('hide.datasets.'+name)
                tree=ast.parse(Path(f'archive/original/dataeval/{name}.py').read_text())
                nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['sample_to_prompt','_generate_config']]
                original={'get_fs_samples_prompt':lambda:''}
                exec(compile(ast.Module(body=nodes,type_ignores=[]),'original_dataset','exec'),original)
                self.assertEqual(module._generate_config(Tokenizer()), original['_generate_config'](Tokenizer()))
                if 'sample_to_prompt' in original:
                    sample=dict(story='Some passage.',question='Which answer?',options=['A','B'])
                    self.assertEqual(module.sample_to_prompt(sample),original['sample_to_prompt'](sample))

    def test_race_list_and_serialized_list_schemas(self):
        fake=types.SimpleNamespace(Dataset=object)
        with patch.dict(sys.modules, {'datasets':fake}):
            parse=importlib.import_module('hide.datasets.race').parse_problems
        value=[dict(question='Q', options=['A','B'], answer='A')]
        self.assertEqual(parse(value),value)
        self.assertEqual(parse(repr(value)),value)
        with self.assertRaises(ValueError):parse({'wrong':'schema'})

    def test_squad_first_reference_is_explicitly_returned(self):
        class Data:
            def __init__(self):
                self.rows=[dict(id='1',story='Context',question='Question?',answer={'text':'First answer'},additional_answers=['Other'])]
            def map(self, fn, **kwargs):
                # Like a mapper that retains only the returned changes, not input mutation.
                self.rows=[dict(row, **fn(copy.deepcopy(row))) for row in self.rows]
                return self
            def set_format(self, **kwargs): pass
        fake=types.SimpleNamespace(Dataset=object,load_from_disk=lambda path:Data())
        with patch.dict(sys.modules,{'datasets':fake}):
            module=importlib.import_module('hide.datasets.SQuAD')
            with patch.object(module,'datasets',fake),patch.object(module,'_save_dataset',return_value='/fake'):
                data=module.get_dataset(lambda prompt,**kw:dict(input_ids=[1,2],attention_mask=[1,1]))
        self.assertEqual(data.rows[0]['answer'],'First answer')
        self.assertEqual(data.rows[0]['prompt'],'Context Q: Question? A:')


if __name__=='__main__':unittest.main()
