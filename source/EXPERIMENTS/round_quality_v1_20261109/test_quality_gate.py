import unittest,json,copy
from quality_gate import select,digest,promotion
class QualityTests(unittest.TestCase):
 def setUp(self):
  self.c={'parent_sha256':'p','learning_businesses':['a','b'],'calibration_businesses':['c'],'holdout_businesses':['h'],'max_per_kind':5,'min_per_kind':2,'min_businesses':2,**{k:k for k in ('source_manifest_sha256','split_manifest_sha256','verifier_sha256','scorer_sha256')}}
  self.rows=[];self.ev={}
  for i in range(4):
   plan={'metric':'revenue'};row={'id':str(i),'prompt':'synthetic '+str(i),'response':json.dumps(plan),'kind':'correction' if i%2==0 else 'replay','split':'learning','business_id':'a' if i<2 else 'b','parent_sha256':'p','target_verified':True};self.rows.append(row)
   self.ev[str(i)]={**{k:self.c[k] for k in ('source_manifest_sha256','split_manifest_sha256','verifier_sha256','scorer_sha256')},**{k:True for k in ('schema','semantic','execution','scorer','blind_prediction_persisted','gold_after_prediction')},'plan_sha256':digest(plan),'prompt_sha256':digest(row['prompt']),'parent_sha256':'p','first_pass_verified_correct':i%2==1}
 def test_balanced(self):
  rows,report=select(self.rows,self.ev,self.c);self.assertEqual(report['rows'],4);self.assertEqual([r['kind'] for r in rows],['correction','replay']*2)
 def test_each_gate_fails_closed(self):
  for key in ('schema','semantic','execution','scorer','blind_prediction_persisted','gold_after_prediction'):
   with self.subTest(key=key):
    ev=copy.deepcopy(self.ev);ev['0'][key]=False
    with self.assertRaises(ValueError):select(self.rows,ev,self.c)
 def test_hash_drift(self):
  for key in ('source_manifest_sha256','split_manifest_sha256','verifier_sha256','scorer_sha256','plan_sha256','prompt_sha256','parent_sha256'):
   with self.subTest(key=key):
    ev=copy.deepcopy(self.ev);ev['0'][key]='changed'
    with self.assertRaises(ValueError):select(self.rows,ev,self.c)
 def test_wrong_split(self):
  self.rows[0]['business_id']='c'
  with self.assertRaises(ValueError):select(self.rows,self.ev,self.c)
 def test_wrong_role(self):
  self.ev['0']['first_pass_verified_correct']=True
  with self.assertRaises(ValueError):select(self.rows,self.ev,self.c)
 def test_no_correction(self):
  self.rows=self.rows[1::2]
  with self.assertRaises(ValueError):select(self.rows,self.ev,self.c)
 def test_duplicate(self):
  self.rows.append(self.rows[0])
  with self.assertRaises(ValueError):select(self.rows,self.ev,self.c)
 def test_diversity(self):
  self.c['min_businesses']=3
  with self.assertRaises(ValueError):select(self.rows,self.ev,self.c)
 def test_promotion_regression(self):
  before={'manifest_sha256':'m','decoder_sha256':'d','scorer_sha256':'s','verdicts':{'1':True,'2':False},'metric':0.0};after={**before,'verdicts':{'1':False,'2':True},'metric':0.2,'gates_passed':True};c={'calibration_manifest_sha256':'m','max_regressions':0,'minimum_improvement':0.01};self.assertFalse(promotion(before,after,c)['promote'])
 def test_promotion_split_drift(self):
  b={'manifest_sha256':'m'}
  with self.assertRaises(ValueError):promotion(b,{'manifest_sha256':'x'},{'calibration_manifest_sha256':'m'})
