import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from hide.checkpoints import catalog, prepare, validate
from hide.provenance import checkpoint_identity


class CheckpointTests(unittest.TestCase):
    def fixture(self,path,kind='llama'):
        path.mkdir(parents=True,exist_ok=True)
        (path/'config.json').write_text(json.dumps({'model_type':kind}))
        (path/'tokenizer.json').write_text('{}')
        (path/'model.safetensors').write_bytes(b'fake test weight bytes, not a trained model')
        return path

    def test_local_checkpoint_is_reused_without_network_or_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); local=self.fixture(root/'models'/'Meta-Llama-3-8B')
            before={p.name:p.read_bytes() for p in local.iterdir()}
            with patch.dict(os.environ,{'HIDE_CHECKPOINT_CACHE':str(root/'cache')}),patch('huggingface_hub.snapshot_download') as download:
                path,record=prepare('llama3-8b',root/'models')
            self.assertEqual(path,local.resolve())
            self.assertEqual(record['origin'],'local')
            self.assertIsNone(record['revision'])
            self.assertTrue(record['identity']['weights_content_hashed'])
            download.assert_not_called()
            self.assertEqual(before,{p.name:p.read_bytes() for p in local.iterdir()})

    def test_missing_base_downloads_exact_revision_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root/'models'/'gemma-2-27b-it','gemma2')
            def download(**kwargs):
                self.fixture(Path(kwargs['local_dir']),'gemma2')
                return kwargs['local_dir']
            with patch.dict(os.environ,{'HIDE_CHECKPOINT_CACHE':str(root/'cache')}),patch('huggingface_hub.snapshot_download',side_effect=download) as mocked:
                path,record=prepare('gemma-2-27b',root/'models')
                again,_=prepare('gemma-2-27b',root/'models')
                self.assertEqual(path,again)
                self.assertEqual(mocked.call_count,1)
                self.assertEqual(mocked.call_args.kwargs['repo_id'],'google/gemma-2-27b')
                self.assertEqual(mocked.call_args.kwargs['revision'],catalog()['gemma-2-27b']['revision'])
                self.assertEqual(record['origin'],'download-cache')
            self.assertFalse((root/'models'/'gemma-2-27b').exists())

    def test_invalid_local_checkpoint_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); local=root/'Meta-Llama-3-8B'; local.mkdir()
            with patch('huggingface_hub.snapshot_download') as download:
                with self.assertRaisesRegex(ValueError,'config.json'): prepare('llama3-8b',root)
                download.assert_not_called()
            self.fixture(local)
            (local/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'absent.safetensors'}}))
            with self.assertRaisesRegex(ValueError,'missing/empty weight'): validate(local,catalog()['llama3-8b'])
            (local/'model.safetensors.index.json').unlink()
            (local/'config.json').write_text(json.dumps({'model_type':'llama','quantization_config':{'bits':4}}))
            with self.assertRaisesRegex(ValueError,'quantized'): validate(local,catalog()['llama3-8b'])

    def test_timing_startup_prepares_only_generator_and_keyword_model(self):
        from argparse import Namespace
        from hide.checkpoints import prepare_arguments
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args=Namespace(model_name='gemma-2-27b',mode='timing',model_path=str(root/'gemma-2-27b'),
                           keyword_model=str(root/'all-MiniLM-L6-v2'),judge_model=str(root/'nli-roberta-large'),
                           output=str(root/'timing.jsonl'))
            def prepared(name,local_root):
                return root/name,{'name':name,'origin':'test'}
            with patch('hide.checkpoints.prepare',side_effect=prepared) as mocked:
                result=prepare_arguments(args)
            self.assertEqual([call.args[0] for call in mocked.call_args_list],['gemma-2-27b','keyword'])
            self.assertEqual(set(result),{'generator','keyword'})
            self.assertTrue((root/'timing.checkpoints.json').is_file())

    def test_weight_hashes_are_cached_and_invalidated_on_change(self):
        from hide.provenance import file_sha256
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); local=self.fixture(root/'model')
            with patch.dict(os.environ,{'HIDE_CHECKPOINT_CACHE':str(root/'cache')}):
                first=checkpoint_identity(local,True)
                with patch('hide.provenance.file_sha256',wraps=file_sha256) as hashed:
                    second=checkpoint_identity(local,True)
                    self.assertFalse(any(Path(call.args[0]).suffix=='.safetensors' for call in hashed.call_args_list))
                self.assertEqual(first,second)
                (local/'model.safetensors').write_bytes(b'changed bytes')
                unverified=checkpoint_identity(local,False)
                self.assertFalse(unverified['weights_content_hashed'])
                fresh=checkpoint_identity(local,True)
                self.assertNotEqual(first['files'],fresh['files'])

    def test_identical_copies_merge_despite_different_timestamps(self):
        from hide.parts import identity
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); first=self.fixture(root/'first')
            second=root/'second'; shutil.copytree(first,second)
            os.utime(second/'model.safetensors',ns=(1,2))
            with patch.dict(os.environ,{'HIDE_CHECKPOINT_CACHE':str(root/'cache')}):
                a=checkpoint_identity(first,True); b=checkpoint_identity(second,True)
            def metadata(checkpoint):
                return {'arguments':{},'checkpoint_identity':{'generator':checkpoint},'source_sha256':{}}
            self.assertEqual(identity(metadata(a)),identity(metadata(b)))

    def test_sentence_transformer_pooling_is_checked_and_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=self.fixture(root/'encoder','bert')
            with self.assertRaisesRegex(ValueError,'modules.json'): validate(path,catalog()['keyword'])
            (path/'modules.json').write_text(json.dumps([
                dict(path='',type='sentence_transformers.models.Transformer'),
                dict(path='1_Pooling',type='sentence_transformers.models.Pooling'),
                dict(path='2_Normalize',type='sentence_transformers.models.Normalize')]))
            pool=path/'1_Pooling'; pool.mkdir(); (pool/'config.json').write_text('{}')
            validate(path,catalog()['keyword'])
            with patch.dict(os.environ,{'HIDE_CHECKPOINT_CACHE':str(root/'cache')}):
                saved=checkpoint_identity(path,True)
            self.assertIn('1_Pooling/config.json',[r['file'] for r in saved['files']])

    def test_offline_missing_checkpoint_and_failed_download_leave_no_ready_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.dict(os.environ,{'HIDE_CHECKPOINT_CACHE':str(root/'cache')}),patch('huggingface_hub.snapshot_download',side_effect=OSError('test connection failure')):
                with self.assertRaises(FileNotFoundError): prepare('gemma-2-27b',root/'models',False)
                with self.assertRaisesRegex(RuntimeError,'huggingface-cli login'): prepare('gemma-2-27b',root/'models',True)
            self.assertFalse(list((root/'cache').rglob('*.ready.json')))


if __name__=='__main__': unittest.main()
