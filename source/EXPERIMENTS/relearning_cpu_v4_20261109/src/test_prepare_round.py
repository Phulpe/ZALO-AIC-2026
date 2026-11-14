import unittest,ast
from pathlib import Path
class ReadyTests(unittest.TestCase):
 def test_no_optimizer_in_collection(self):
  source=Path(__file__).with_name('prepare_round.py').read_text();ast.parse(source)
  self.assertNotIn('.backward(',source);self.assertNotIn('train_warm',source)
  self.assertIn('verify_parent(',source);self.assertIn('is_trainable=False',source)
 def test_selection_independent_of_gold(self):
  source=Path(__file__).with_name('learning_probe.py').read_text()
  self.assertIn("supported={q['id'] for q in pool if canonical_total(q) is not None}",source)
  self.assertIn('LearningRound(',source)
