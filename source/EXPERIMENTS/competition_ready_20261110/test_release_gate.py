import unittest, tempfile, importlib.util, hashlib
from pathlib import Path
root=Path(__file__).parent
spec=importlib.util.spec_from_file_location('docker_entrypoint',root/'DOCKER_DRAFT/entrypoint.py')
entry=importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class ReleaseGateTests(unittest.TestCase):
    def test_unapproved_release_is_blocked(self):
        with self.assertRaises(ValueError):
            entry.verify_release(root, {'ready_to_infer':False})

    def test_model_choice_cannot_be_implicit(self):
        with self.assertRaises(ValueError):
            entry.verify_release(root, {'ready_to_infer':True,'files':{'x':'x'}})

    def test_changed_weight_or_source_is_blocked(self):
        with tempfile.TemporaryDirectory(dir=root) as d:
            p=Path(d);(p/'fixture').write_bytes(b'changed')
            manifest={'ready_to_infer':True,'model_selection':'candidate',
                'files':{'fixture':hashlib.sha256(b'original').hexdigest()}}
            with self.assertRaises(ValueError):entry.verify_release(p,manifest)

    def test_inventory_cannot_escape_root(self):
        with tempfile.TemporaryDirectory(dir=root) as d:
            p=Path(d);manifest={'ready_to_infer':True,'model_selection':'candidate',
                'files':{'../outside':'0'*64}}
            with self.assertRaises(ValueError):entry.verify_release(p,manifest)


if __name__=='__main__':unittest.main()
