import unittest
from runtime_gpu import single_gpu_env,require_single_gpu

class GpuTests(unittest.TestCase):
    def test_child_visibility_is_one_physical_gpu(self):
        original={'CUDA_VISIBLE_DEVICES':'0,1','PYTHONPATH':'official-deps'}
        child=single_gpu_env(original)
        self.assertEqual(child['CUDA_VISIBLE_DEVICES'],'0')
        self.assertEqual(child['PYTHONPATH'],'official-deps')
        self.assertEqual(original['CUDA_VISIBLE_DEVICES'],'0,1')
    def test_multi_gpu_rejected_before_model_load(self):
        for n in [0,2,4]:
            with self.assertRaises(RuntimeError):require_single_gpu(n)
    def test_single_gpu_allowed(self):require_single_gpu(1)
