import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from fit_abstention import wilson_lower
from abstention_policy import apply_policy,signature


class PolicyTests(unittest.TestCase):
 def test_no_observations_not_evidence(self):self.assertEqual(wilson_lower(0,0),0)
 def test_coin_flip_has_no_positive_lower_bound(self):self.assertLess(wilson_lower(50,100),0.5)
 def test_small_perfect_sample_is_not_certainty(self):self.assertLess(wilson_lower(1,1),0.5)
 def test_strong_group_can_pass(self):self.assertGreater(wilson_lower(18,20),0.5)
 def test_rejection_never_edits_numeric_answer(self):
  body={'type':'number','answer':123};result=apply_policy({'metric':'revenue','group_by':None},body,{'allowed_signatures':[]})
  self.assertEqual(result['type'],'abstain');self.assertEqual(body,{'type':'number','answer':123})
 def test_permitted_prediction_unchanged(self):
  plan={'metric':'revenue','group_by':None};body={'type':'number','answer':123}
  self.assertEqual(apply_policy(plan,body,{'allowed_signatures':[signature(plan,body)]}),body)
 def test_proven_noncoverage_kept(self):self.assertEqual(apply_policy({}, {'type':'unanswerable'},{'allowed_signatures':[]}),{'type':'unanswerable'})
 def test_raw_evaluation_has_no_policy(self):self.assertEqual(apply_policy({}, {'type':'number','answer':1},None),{'type':'number','answer':1})


if __name__=='__main__':unittest.main()
