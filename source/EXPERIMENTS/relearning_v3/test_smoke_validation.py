import unittest
from pathlib import Path
from unittest.mock import patch,Mock
from smoke_validation import tensor_delta,validate_after_training
class SmokeTests(unittest.TestCase):
    def test_finite_update(self):
        r=tensor_delta({'a':[1.,2.]},{'a':[1.,2.1]});self.assertEqual(r['changed_elements'],1);self.assertGreater(r['delta_l2'],0)
    def test_unchanged_rejected(self):
        with self.assertRaises(ValueError):tensor_delta({'a':[1.]},{'a':[1.]})
    def test_nan_rejected(self):
        with self.assertRaises(ValueError):tensor_delta({'a':[1.]},{'a':[float('nan')]})
    def test_wrong_keys_rejected(self):
        with self.assertRaises(ValueError):tensor_delta({'a':[1.]},{'b':[1.]})
    def test_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):tensor_delta({'a':[1.]},{'a':[1.,2.]})
    def test_reload_receipt_requires_real_flag_and_loss(self):
        with patch('smoke_validation.safe_adapter_tensors',side_effect=[{'a':[1.]},{'a':[2.]}]):
            with self.assertRaises(ValueError):validate_after_training(Path('parent'),Path('run/checkpoint'),lambda p:'sha',lambda p:{'loss':float('inf'),'mask_shift_verified':True,'reloaded_from_saved_adapter':True})
    def test_valid_receipt_does_not_claim_full_resume(self):
        with patch('smoke_validation.safe_adapter_tensors',side_effect=[{'a':[1.]},{'a':[2.]}]):
            result=validate_after_training(Path('parent'),Path('run/checkpoint'),lambda p:'sha',lambda p:{'loss':1.,'mask_shift_verified':True,'reloaded_from_saved_adapter':True})
        self.assertFalse(result['checkpoint_full_resume_verified']);self.assertEqual(result['status'],'passed')
    def test_bad_manifest_stops_before_reload(self):
        reload=Mock()
        with self.assertRaises(ValueError):validate_after_training(Path('parent'),Path('run/checkpoint'),Mock(side_effect=ValueError()),reload)
        reload.assert_not_called()
if __name__=='__main__':unittest.main()
