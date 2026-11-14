"""Synthetic failure-path verification of sync commits; no Kaggle network calls."""
from pathlib import Path
import json
import tempfile
import unittest
from run_sync import seal, import_saved_run, verify, digest


class SyncTests(unittest.TestCase):
    def make_run(self, root, run_id='r1'):
        root.mkdir()
        (root / 'weights.bin').write_bytes(b'synthetic adapter bytes')
        h = seal(root, {'run_id':run_id,'action':'train','state':'completed',
                        'created_utc':'2026-11-07T00:00:00+00:00','metrics':{}})
        return {'saved_version_confirmed':True,
            'version_url':'https://www.kaggle.com/code/test/test?scriptVersionId=1',
            'remote_manifest_sha256':h}

    def test_commit_only_verified_and_best_separate(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t); receipt = self.make_run(p/'source')
            import_saved_run(p/'source',p/'runs',receipt)
            state = json.loads((p/'runs/SYNC_STATE.json').read_text())
            self.assertEqual(state['LATEST_SYNCED'],'r1')
            self.assertIsNone(state['BEST_MODEL'])

    def test_missing_file_keeps_previous_pointer(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); receipt=self.make_run(p/'source')
            import_saved_run(p/'source',p/'runs',receipt)
            old=(p/'runs/SYNC_STATE.json').read_bytes()
            receipt2=self.make_run(p/'second','r2')
            (p/'second/weights.bin').unlink()
            with self.assertRaises(ValueError): import_saved_run(p/'second',p/'runs',receipt2)
            self.assertEqual((p/'runs/SYNC_STATE.json').read_bytes(),old)

    def test_remote_manifest_mismatch_never_commits(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); receipt=self.make_run(p/'source');receipt['remote_manifest_sha256']='0'*64
            with self.assertRaises(ValueError): import_saved_run(p/'source',p/'runs',receipt)
            self.assertFalse((p/'runs/SYNC_STATE.json').exists())

    def test_existing_conflict_preserves_original(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); receipt=self.make_run(p/'source')
            import_saved_run(p/'source',p/'runs',receipt)
            old=(p/'runs/r1/weights.bin').read_bytes()
            (p/'source/weights.bin').write_bytes(b'conflict')
            meta=json.loads((p/'source/RUN_METADATA.json').read_text())
            receipt['remote_manifest_sha256']=seal(p/'source',meta)
            with self.assertRaises(ValueError):import_saved_run(p/'source',p/'runs',receipt)
            self.assertEqual((p/'runs/r1/weights.bin').read_bytes(),old)

    def test_unsaved_version_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); receipt=self.make_run(p/'source');receipt['saved_version_confirmed']=False
            with self.assertRaises(ValueError):import_saved_run(p/'source',p/'runs',receipt)


if __name__ == '__main__':unittest.main()
