import ast,unittest
from pathlib import Path
class LoaderProbeTests(unittest.TestCase):
 def test_probe_before_model_load(self):
  s=Path(__file__).with_name('public_only_811.py').read_text()
  self.assertLess(s.index("os.environ['OMP_NUM_THREADS']"),s.index('from workflow_811 import'))
  self.assertLess(s.index('probe(Path(sys.argv[1]))'),s.index('main(sys.argv[1])'))
 def test_probe_has_no_training_or_pickle(self):
  s=Path(__file__).with_name('loader_probe.py').read_text();ast.parse(s)
  self.assertNotIn('torch.load(',s);self.assertNotIn('.backward(',s)
  self.assertIn('torch.set_num_threads(1)',s)
