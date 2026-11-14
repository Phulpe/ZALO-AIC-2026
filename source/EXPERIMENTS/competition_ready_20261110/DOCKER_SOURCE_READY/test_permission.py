import unittest
from permission import permitted
class PermissionTests(unittest.TestCase):
    def test_unverified_rejected(self):self.assertFalse(permitted({'status':'unverified'}))
    def test_attestation_without_acceptance_rejected(self):self.assertFalse(permitted({'status':'user_attested','user_statement':'allowed','agents_sha256':'hash'}))
    def test_explicit_attestation_accepted_without_false_verification(self):
        e={'status':'user_attested','accepted_for_this_run':True,'user_statement':'allowed','agents_sha256':'hash','independently_verified':False}
        self.assertTrue(permitted(e));self.assertFalse(e['independently_verified'])
