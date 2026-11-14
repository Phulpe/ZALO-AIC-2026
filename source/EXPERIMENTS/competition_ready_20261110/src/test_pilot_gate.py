import unittest,tempfile,json
from pathlib import Path
from pilot_gate import after_training,learning_gate
from relearning import LearningRound

def summary(gain=1,lost=0):return {'mode':'evaluation','weights_updated':False,'expected':4,'attempted':4,'real_output_ids':4,'correction_before_correct':0,'correction_after_correct':gain,'correction_total':2,'replay_before_correct':2,'replay_after_correct':2-lost,'correct_to_wrong':lost}

class GateTests(unittest.TestCase):
 def test_zero_correction_blocks_calibration_and_public(self):
  called=[];r=after_training({'run_full_calibration':True,'run_public':True},lambda:summary(0),lambda:called.append('calibration'),lambda:called.append('public'));self.assertFalse(r['promoted']);self.assertEqual(called,[])
 def test_forgetting_blocks(self):self.assertFalse(learning_gate(summary(2,1))['passed'])
 def test_post_before_calibration_before_public(self):
  called=[]
  def post():called.append('post');return summary()
  def cal():called.append('cal');return {'promote':True}
  r=after_training({'run_full_calibration':True,'run_public':True},post,cal,lambda:called.append('public'));self.assertEqual(called,['post','cal','public']);self.assertTrue(r['public_ran'])
 def test_public_never_runs_for_rejected_candidate(self):
  called=[];r=after_training({'run_full_calibration':True,'run_public':True},summary,lambda:{'promote':False},lambda:called.append('public'));self.assertFalse(r['public_ran']);self.assertFalse(called)
 def test_no_promotion_from_learning_only(self):
  r=after_training({'run_full_calibration':False},summary);self.assertTrue(r['learning_gate']['passed']);self.assertFalse(r['promoted'])
 def test_missing_predictions_and_invalid_counts_fail(self):
  for change in [{'attempted':3},{'real_output_ids':3},{'weights_updated':True},{'correction_after_correct':3},{'correct_to_wrong':1},{'mode':'learning'}]:
   with self.subTest(change=change),self.assertRaises(ValueError):learning_gate({**summary(),**change})

class BudgetLedgerTests(unittest.TestCase):
 def test_stop_before_next_question_preserves_pending_and_denominator(self):
  with tempfile.TemporaryDirectory() as tmp:
   pool=[{'id':str(i),'split':'learning','business_id':'b','prompt':'synthetic','source_ref':'b','as_of_date':'2024-03-01'} for i in range(3)];gold_calls=[]
   def gold(i):gold_calls.append(i);return 1
   config={'mode':'learning','attempts_per_question':1,'learning_businesses':['b'],'calibration_businesses':[],'holdout_businesses':[]}
   obj=LearningRound(tmp,config,pool,gold,lambda *a:{},lambda *a:1,lambda p:None,lambda *a:True,lambda a,b:a==b)
   metrics,_,_=obj.process(stop_requested=lambda:bool(gold_calls));self.assertEqual(gold_calls,['0']);self.assertEqual(metrics['attempted'],1);self.assertEqual(metrics['pending'],2);self.assertEqual(metrics['attempt_coverage'],1/3);self.assertIsNone(metrics['verified_correction_coverage']);self.assertEqual(obj.ledger['questions']['1']['attempts'],[])

if __name__=='__main__':unittest.main()
