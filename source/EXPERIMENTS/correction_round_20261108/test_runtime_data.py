import tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
from runtime_data import LazyGoldReader,pending_coverage,reconcile_coverage
from relearning import sha
class RuntimeDataTests(unittest.TestCase):
    def test_gold_not_opened_at_construction(self):
        with patch.object(Path,'open',side_effect=AssertionError('Gold opened early')):LazyGoldReader('unused',{'x'},'hash')
    def test_wrong_split_rejected_before_open(self):
        with patch.object(Path,'open',side_effect=AssertionError('Gold opened')):
            with self.assertRaises(ValueError):LazyGoldReader('unused',{'x'},'hash')('calibration_id')
    def test_gold_hash_checked(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'gold';p.write_text(json.dumps({'id':'x','answer':1})+'\n')
            with self.assertRaises(ValueError):LazyGoldReader(p,{'x'},'wrong')('x')
            self.assertEqual(LazyGoldReader(p,{'x'},sha(p))('x')['answer'],1)
    def test_partial_coverage_never_claims_full_pool(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'coverage';pending_coverage(p,[{'id':str(i)} for i in range(4)],'parent','split')
            m=reconcile_coverage(p,{'questions':{'0':{'status':'resolved','attempts':[{'state':'correct'}]}}})
            self.assertEqual(m['attempt_coverage'],.25);self.assertEqual(m['pending'],3)
    def test_unstarted_attempt_not_counted(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'coverage';pending_coverage(p,[{'id':'x'}],'parent','split')
            m=reconcile_coverage(p,{'questions':{'x':{'status':'pending','attempts':[{'state':'started'}]}}})
            self.assertEqual(m['attempted'],0)
    def test_coverage_unknown_id_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'coverage';pending_coverage(p,[{'id':'x'}],'parent','split')
            with self.assertRaises(ValueError):reconcile_coverage(p,{'questions':{'other':{'status':'resolved','attempts':[]}}})
