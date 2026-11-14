import unittest,tempfile,json,copy
from pathlib import Path
from unittest.mock import patch
from protected_regression import evaluate_protected
from relearning import atomic,seal
from quality_gate import digest
from pilot_gate import learning_gate

class FakeBridge:
 calls=[];wrong=set()
 def __init__(self,*args,**kwargs):pass
 def predict(self,q,n):
  self.calls.append(q['id']);return {'correct':q['id'] not in self.wrong}
 def semantic(self,q,p):return True
 def execute(self,q,p):return {'type':'number','answer':100 if p['correct'] else 0}

class ProtectedRegressionTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.bank=self.root/'bank';self.bank.mkdir();self.out=self.root/'out';self.rows=[];self.pool=[]
  for kind,n in [('correction',2),('replay',3)]:
   for i in range(n):
    key=kind+str(i);self.rows.append({'id':key,'kind':kind,'business_id':'biz','split':'learning','parent_sha256':'p','target_verified':True,'prompt':'synthetic '+key,'response':'{}'});self.pool.append({'id':key,'business_id':'biz','split':'learning','source_ref':'biz','prompt':'synthetic '+key})
  self.reference=copy.deepcopy(self.rows[:2]);FakeBridge.calls=[];FakeBridge.wrong=set();self.write()
 def write(self):
  atomic(self.bank/'PRIVATE_SELECTED.json',self.rows);atomic(self.bank/'PRIVATE_REGRESSION_BANK.json',[r for r in self.rows if r['kind']=='replay']);atomic(self.bank/'EVIDENCE_LINEAGE.json',{'full_replay_regression_cases':3});self.manifest=seal(self.bank)
 def run_eval(self,**kw):
  def gold(key):
   records=json.loads((self.out/'PRIVATE_PREDICTIONS.json').read_text());self.assertIn('plan',records[key]);return {'type':'number','answer':100}
  with patch('baseline_bridge.BaselineBridge',FakeBridge),patch('protected_regression.BaselineBridge',FakeBridge):
   return evaluate_protected(None,None,self.bank,kw.get('manifest',self.manifest),'p',['biz'],self.pool,gold,{},self.out,kw.get('reference',self.reference),kw.get('reference_hash',digest(self.reference)),critical_count=2)
 def test_full_bank_once_gold_after_persisted_prediction_and_critical_subset(self):
  full,critical=self.run_eval();self.assertEqual(full['expected'],5);self.assertEqual(len(FakeBridge.calls),5);self.assertEqual(len(set(FakeBridge.calls)),5);self.assertEqual(critical['correction_after_correct'],2);self.assertEqual(critical['extra_model_generations'],0);self.assertTrue(learning_gate(full)['passed']);self.assertTrue(learning_gate(critical)['passed'])
 def test_any_correct_to_wrong_blocks_gate(self):
  FakeBridge.wrong={'replay1'};full,_=self.run_eval();self.assertEqual(full['correct_to_wrong'],1);self.assertFalse(learning_gate(full)['passed'])
 def test_old_critical_failure_cannot_be_hidden_by_new_corrections(self):
  extra=copy.deepcopy(self.rows[0]);extra.update(id='new_correction',prompt='synthetic new');self.rows.append(extra);self.pool.append({'id':extra['id'],'business_id':'biz','split':'learning','source_ref':'biz','prompt':extra['prompt']});self.write()
  FakeBridge.wrong={'correction0','correction1'};full,critical=self.run_eval();self.assertTrue(learning_gate(full)['passed']);self.assertFalse(learning_gate(critical)['passed'])
 def test_missing_critical_case_blocks_before_model(self):
  other=copy.deepcopy(self.reference);other[0]['id']='absent'
  with self.assertRaisesRegex(ValueError,'critical corrections'):self.run_eval(reference=other,reference_hash=digest(other))
  self.assertFalse(FakeBridge.calls)
 def test_manifest_mismatch_blocks_before_model(self):
  with self.assertRaises(ValueError):self.run_eval(manifest='bad')
  self.assertFalse(FakeBridge.calls)
 def test_critical_reference_mismatch_blocks_before_model(self):
  with self.assertRaises(ValueError):self.run_eval(reference_hash='bad')
  self.assertFalse(FakeBridge.calls)
 def test_wrong_split_blocks_before_gold_or_generation(self):
  self.rows[-1]['business_id']='calibration';self.write()
  with self.assertRaises(ValueError):self.run_eval()
  self.assertFalse(FakeBridge.calls)
 def test_duplicate_role_ids_blocks_before_model(self):
  self.rows[-1]['id']=self.rows[0]['id'];self.write()
  with self.assertRaises(ValueError):self.run_eval()
  self.assertFalse(FakeBridge.calls)

if __name__=='__main__':unittest.main()
