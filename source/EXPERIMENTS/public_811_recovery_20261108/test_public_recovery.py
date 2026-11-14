from pathlib import Path
import unittest,hashlib
from public_only_811 import CANDIDATE,MANIFEST,validate_candidate

class RecoveryTests(unittest.TestCase):
    def test_companion_audit_file_is_packaged_and_identical(self):
        root=Path(__file__).parent
        self.assertEqual(hashlib.sha256((root/'audit_output.py').read_bytes()).hexdigest(),hashlib.sha256((root/'baseline/tools/audit_output.py').read_bytes()).hexdigest())
    def test_verified_candidate_allowed(self):validate_candidate({'steps':4,'synthetic_training':False,'adapter_sha256':CANDIDATE,'manifest_sha256':MANIFEST})
    def test_other_weights_steps_and_synthetic_smoke_rejected(self):
        for key,value in [('steps',1),('synthetic_training',True),('adapter_sha256','parent'),('manifest_sha256','unverified')]:
            receipt={'steps':4,'synthetic_training':False,'adapter_sha256':CANDIDATE,'manifest_sha256':MANIFEST};receipt[key]=value
            with self.assertRaises(ValueError):validate_candidate(receipt)
