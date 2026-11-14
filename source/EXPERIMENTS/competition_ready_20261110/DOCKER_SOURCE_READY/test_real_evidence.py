import unittest,tempfile,json
from pathlib import Path
from relearning import LearningRound
from baseline_bridge import canonical_total,canonical_correction,BaselineBridge,matches
from real_evidence import export
from test_core import fixture
class IntegrationTests(unittest.TestCase):
 def test_blind_to_correction_to_quality_dataset(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp);pool=[]
   for i,month in enumerate([1,2]):
    ctx={'question':'Tong doanh thu thang '+str(month)+' nam 2026 la bao nhieu?','as_of_date':'2026-02-28'}
    pool.append({'id':str(i),'prompt':json.dumps(ctx),'as_of_date':ctx['as_of_date'],'business_id':'a','source_ref':'a','split':'learning'})
   bridge=BaselineBridge(None,None,{'a':fixture()});gold={q['id']:bridge.execute(q,canonical_total(q)) for q in pool}
   cfg={'mode':'learning','attempts_per_question':1,'learning_businesses':['a'],'calibration_businesses':['b'],'holdout_businesses':['c'],'parent_sha256':'p'}
   def predict(q,n):
    plan=canonical_total(q)
    return {**plan,'metric':'orders'} if q['id']=='0' else plan
   def reader(i):
    ledger=json.loads((path/'ledger.json').read_text());assert ledger['questions'][i]['attempts'][0]['state'] in ('predicted','invalid_plan','correct','wrong');return gold[i]
   from planner import validate_plan
   metrics,replay,correction=LearningRound(path,cfg,pool,reader,predict,bridge.execute,validate_plan,bridge.semantic,matches,canonical_correction).process()
   policy={**cfg,'min_per_kind':1,'max_per_kind':2,'min_businesses':1,**{k:k for k in ('source_manifest_sha256','split_manifest_sha256','verifier_sha256','scorer_sha256')}}
   result=export(path,pool,bridge,reader,policy)
   self.assertEqual(result['state'],'passed');self.assertEqual(result['rows'],2);self.assertEqual(len(correction),1);self.assertEqual(len(replay),1)
