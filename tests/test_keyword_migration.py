import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_parts as parts_tests
from hide.export_results import export, inspect_run, restore
from hide.parts import (compatible_detection_sources, comparison_identity, keyword_fix_policy,
                        load_plan, merge, part_path, plan_digest, portable_sources)
from hide.provenance import atomic_json, file_sha256, source_files

spec = importlib.util.spec_from_file_location('keyword_migration', Path(__file__).resolve().parents[1] / 'scripts/migrate_keyword_fallback.py')
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


class KeywordMigrationTests(unittest.TestCase):
    def test_real_policy_is_narrowly_pinned_and_rejects_timing(self):
        policy = keyword_fix_policy()
        current = portable_sources(source_files())
        legacy = policy['legacy_source_sha256']
        self.assertTrue(compatible_detection_sources(legacy, current))
        altered = dict(current, **{'hide__runner.py': 'changed'})
        self.assertFalse(compatible_detection_sources(legacy, altered))
        self.assertFalse(compatible_detection_sources(dict(legacy, extra='changed'), current))
        self.assertFalse(compatible_detection_sources(legacy, dict(current, **{'hide__core.py': 'changed'})))
        legacy_paths = {'/original/'+'/'.join(key.split('__', 1)): value for key, value in legacy.items()}
        accepted = comparison_identity({'arguments': {'mode': 'detection'}, 'source_sha256': legacy_paths}, {'source_sha256': current})
        self.assertEqual(accepted['source_sha256'], current)
        with self.assertRaisesRegex(ValueError, 'Unapproved source'):
            comparison_identity({'arguments': {'mode': 'timing'}, 'source_sha256': legacy_paths}, {'source_sha256': current})

    def test_migration_merge_export_preserve_original_sources_and_failed_bytes(self):
        fixture = parts_tests.PartTests()
        with tempfile.TemporaryDirectory() as directory, fixture.small_plan(), patch.dict('hide.parts.DATASET_COUNTS', {'nq_open': 5}):
            root = Path(directory)
            _, plan, old_queue = fixture.fixture(root)
            original_files = {}
            legacy_text = b'# Synthetic previous core for provenance testing\n'
            legacy = dict(plan['source_sha256'], **{'hide__core.py': hashlib.sha256(legacy_text).hexdigest()})
            policy = dict(keyword_fix_policy(), legacy_source_sha256=legacy)
            plan['source_sha256'] = legacy
            plan['sha256'] = plan_digest(plan)
            atomic_json(old_queue/'plan.json', plan)
            for task in plan['tasks']:
                path = part_path(old_queue, task)
                original_files[task['id']] = path.read_bytes()
                meta_path = path.with_suffix('.manifest.json')
                meta = json.loads(meta_path.read_text())
                for key in meta['source_sha256']:
                    if key.endswith('/hide/core.py'):
                        meta['source_sha256'][key] = legacy['hide__core.py']
                atomic_json(meta_path, meta)
                (path.with_suffix('.sources')/'hide__core.py').write_bytes(legacy_text)
            failed_task = plan['tasks'][-1]
            failed = part_path(old_queue, failed_task)
            error_bytes = b'{"id":"4","status":"error","error":"Output keyword extraction failed"}\n'
            failed.write_bytes(error_bytes)
            atomic_json(failed.with_suffix('.failed.json'), {'error': 'failed'})
            old_hashes = {str(p.relative_to(old_queue)): file_sha256(p) for p in old_queue.rglob('*') if p.is_file()}
            with patch('hide.parts.keyword_fix_policy', return_value=policy), patch.object(migration, 'keyword_fix_policy', return_value=policy):
                new_queue = root/'fixed'
                record = migration.migrate(old_queue, new_queue)
                self.assertEqual(len(record['retained_parts']), 2)
                self.assertEqual(len(record['restarted_parts']), 1)
                for rel, digest in old_hashes.items():
                    self.assertEqual(file_sha256(old_queue/rel), digest)
                self.assertFalse(part_path(new_queue, failed_task).exists())
                for task in plan['tasks'][:-1]:
                    old = part_path(old_queue, task)
                    new = part_path(new_queue, task)
                    self.assertEqual(new.read_bytes(), old.read_bytes())
                    self.assertEqual(new.with_suffix('.manifest.json').read_bytes(), old.with_suffix('.manifest.json').read_bytes())
                saved = new_queue/'previous_incomplete_parts'/failed_task['id']/'qa'/failed.with_suffix('.jsonl.bak').name
                self.assertEqual(saved.read_bytes(), error_bytes)
                new_plan = load_plan(new_queue/'plan.json', check_source=True)
                # Simulate the failed part being rerun successfully under the new implementation.
                target = part_path(new_queue, failed_task)
                import shutil
                shutil.copytree(failed.parent.parent, target.parent.parent)
                target.write_bytes(original_files[failed_task['id']])
                target.with_suffix('.failed.json').unlink()
                meta = json.loads(target.with_suffix('.manifest.json').read_text())
                meta['source_sha256'] = source_files()
                atomic_json(target.with_suffix('.manifest.json'), meta)
                for source in source_files():
                    p = Path(source)
                    (target.with_suffix('.sources')/(p.parent.name+'__'+p.name)).write_bytes(p.read_bytes())
                merged_root = root/'merged'
                merge(new_queue/'plan.json', [new_queue], merged_root)
                merged = merged_root/'qa'/'llama3-8b_nq_open.jsonl'
                self.assertTrue(inspect_run(merged)['complete'])
                manifest = json.loads(merged.with_suffix('.manifest.json').read_text())
                self.assertEqual(manifest['source_compatibility_policy'], policy)
                for task in plan['tasks'][:-1]:
                    self.assertEqual(portable_sources(manifest['generation_sources_by_part'][task['id']]), legacy)
                self.assertEqual(portable_sources(manifest['generation_sources_by_part'][failed_task['id']]), new_plan['source_sha256'])
                self.assertIn('merge implementation', manifest['source_role'])
                self.assertEqual((merged_root/'provenance'/'previous_incomplete_parts_0'/failed_task['id']/'qa'/failed.with_suffix('.jsonl.bak').name).read_bytes(), error_bytes)
                export(merged_root, root/'bundle')
                restore(root/'bundle', root/'restored')
                self.assertEqual(merged.read_bytes(), (root/'restored'/'qa'/merged.name).read_bytes())
                with self.assertRaisesRegex(ValueError, 'new queue'):
                    migration.migrate(old_queue, new_queue)
