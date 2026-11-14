import unittest,tempfile,copy,json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from relearning import sha
from source_annotations import original_notes,VerifiedLearningShops
from learning_executor import repair_shop,execute_learning
from failure_triage import classify,plan_difference
from planner import PlanError

def plan(time=None):return {'metric':'revenue','time':time or {'kind':'month','value':'2026-01'},'filters':[],'group_by':None,'order':'desc','limit':None,'unit':None,'answer_type':'number','unresolved':[]}

class AnnotationTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.file=self.root/'data/biz/sales.csv';self.file.parent.mkdir(parents=True)
  self.raw=pd.DataFrame([{'product':'Paper','product_id':'A','date':'2026-01-10','order_id':'O1','qty':2,'price':50,'line_total':100,'unit':'piece'}, {'product':'Ghi chú: tài liệu','product_id':None,'date':None,'order_id':None,'qty':None,'price':None,'line_total':None,'unit':None}]);self.write()
 def write(self):
  self.raw.to_csv(self.file,index=False);self.manifest={'business_source_files':{'data/biz/sales.csv':{'bytes':self.file.stat().st_size,'sha256':sha(self.file)}}}
  parsed=self.raw.copy();parsed.date=pd.to_datetime(parsed.date,errors='coerce');mapping={n:n for n in self.raw};parsed['_source']='sales.csv:'
  lines=parsed.copy();lines['order_id']=lines.order_id.ffill();lines['date']=lines.date.ffill();lines['_status']='completed';lines['base_qty']=lines.qty;lines['base_unit']=lines.unit
  self.shop=SimpleNamespace(tables={'sales':parsed},audit=[{'file':'sales.csv','sheet':'','kind':'sales','mapping':mapping}],lines=lines,orders=pd.DataFrame(),products=lines[['product_id','product']].dropna().copy(),readme='Dữ liệu có đến hết 31/01/2026',end=pd.Timestamp('2026-01-31'),start=pd.Timestamp('2026-01-10'),has_orders=True)
 def proofs(self):return original_notes(self.root,'biz',self.shop,self.manifest,{'biz'})
 def test_raw_source_proves_note_and_executes_without_imputing(self):
  proofs=self.proofs();self.assertEqual(len(proofs[0]),1);clean,report=repair_shop(self.shop,proofs);self.assertEqual(report['removed'],1);self.assertEqual(report['amounts_imputed'],0);self.assertEqual(len(clean.lines),1)
  self.assertEqual(execute_learning(self.shop,'Doanh thu tháng 1 năm 2026','2026-01-31',plan(),'learning',proofs),{'type':'number','answer':100})
 def test_without_source_proof_missing_amount_still_rejected(self):
  self.assertIs(repair_shop(self.shop)[0],self.shop)
  with self.assertRaises(PlanError):execute_learning(self.shop,'Doanh thu tháng 1 năm 2026','2026-01-31',plan(),'learning')
 def test_original_cached_shop_and_source_unchanged(self):
  before=self.shop.lines.copy(deep=True);digest=sha(self.file);repair_shop(self.shop,self.proofs());pd.testing.assert_frame_equal(before,self.shop.lines);self.assertEqual(sha(self.file),digest)
 def test_real_product_named_note_not_removed(self):
  self.raw.loc[1,['date','order_id','qty','price','line_total','unit']]=['2026-01-11','O2',1,20,20,'piece'];self.write();self.assertFalse(self.proofs()[0]);self.assertEqual(len(repair_shop(self.shop,self.proofs())[0].lines),2)
 def test_missing_price_on_real_transaction_is_not_note_proof(self):
  self.raw.loc[1,['date','order_id']]=['2026-01-11','O2'];self.write();self.assertFalse(self.proofs()[0])
 def test_malformed_date_is_not_blank_date(self):
  self.raw.loc[1,'date']='broken date';self.write();self.assertTrue(pd.isna(self.shop.tables['sales'].loc[1,'date']));self.assertFalse(self.proofs()[0])
 def test_unparseable_money_is_not_blank(self):
  self.raw['line_total']=self.raw.line_total.astype(object);self.raw.loc[1,'line_total']='unknown';self.write();self.assertFalse(self.proofs()[0])
 def test_numeric_zero_is_not_blank(self):
  self.raw.loc[1,'qty']=0;self.write();self.assertFalse(self.proofs()[0])
 def test_no_marker_no_drop(self):
  self.raw.loc[1,'product']='Unknown item';self.write();self.assertFalse(self.proofs()[0])
 def test_source_hash_changed_rejected(self):
  self.file.write_text('changed',encoding='utf-8')
  with self.assertRaisesRegex(ValueError,'hash mismatch'):self.proofs()
 def test_wrong_business_rejected(self):
  with self.assertRaisesRegex(ValueError,'outside learning'):original_notes(self.root,'calibration',self.shop,self.manifest,{'biz'})
 def test_wrong_split_rejected_even_if_note_proven(self):
  with self.assertRaisesRegex(ValueError,'evaluation/submission'):execute_learning(self.shop,'x','2026-01-31',plan(),'calibration',self.proofs())
 def test_coverage_guard_preserved(self):
  with self.assertRaisesRegex(PlanError,'predates'):execute_learning(self.shop,'Doanh thu tháng 12 năm 2025','2026-01-31',plan({'kind':'month','value':'2025-12'}),'learning',self.proofs())
 def test_cached_proof_detects_file_change(self):
  data=self.root/'DATA_MANIFEST.json';data.write_text(json.dumps(self.manifest),encoding='utf-8');split=self.root/'SPLIT_MANIFEST.json';split.write_text(json.dumps({'groups':{'learning':['biz']}}),encoding='utf-8');provider=VerifiedLearningShops({'biz':self.shop},self.root,sha(data),sha(split));provider.annotation_proofs('biz',self.shop);self.file.write_text('changed',encoding='utf-8')
  with self.assertRaisesRegex(ValueError,'changed after proof'):provider.annotation_proofs('biz',self.shop)
 def test_no_fake_zero_when_only_annotations_remain(self):
  self.shop.lines=self.shop.lines.iloc[1:].copy()
  with self.assertRaisesRegex(PlanError,'No verified transaction'):repair_shop(self.shop,self.proofs())
 def test_ambiguous_xlsx_header_blocks_proof(self):
  path=self.file.with_suffix('.xlsx');header=list(self.raw);pd.DataFrame([header,header,*self.raw.values.tolist()]).to_excel(path,index=False,header=False);self.shop.audit[0]['file']=path.name;self.shop.audit[0]['sheet']='Sheet1';self.manifest={'business_source_files':{'data/biz/'+path.name:{'bytes':path.stat().st_size,'sha256':sha(path)}}};self.assertFalse(self.proofs()[0])
 def test_source_provider_is_lazy_when_executor_already_succeeds(self):
  self.shop.lines=self.shop.lines.iloc[:1].copy()
  def forbidden():raise AssertionError('Source repair not needed')
  self.assertEqual(execute_learning(self.shop,'Doanh thu tháng 1 năm 2026','2026-01-31',plan(),'learning',forbidden)['answer'],100)
 def test_duplicate_product_label_cannot_identify_annotation(self):
  self.raw.loc[0,'product']=self.raw.loc[1,'product'];self.write();self.assertFalse(self.proofs()[0]);self.assertIs(repair_shop(self.shop,self.proofs())[0],self.shop)
 def test_duplicate_cached_rows_not_deleted_by_one_raw_proof(self):
  self.shop.lines=pd.concat([self.shop.lines,self.shop.lines.iloc[1:]],ignore_index=True);self.assertEqual(repair_shop(self.shop,self.proofs())[1]['removed'],0)
 def test_uncomputed_formula_is_not_proven_empty(self):
  path=self.file.with_suffix('.xlsx');raw=self.raw.copy();raw['line_total']=raw.line_total.astype(object);raw.loc[1,'line_total']='=1+1';raw.to_excel(path,index=False);self.shop.audit[0].update(file=path.name,sheet='Sheet1');self.manifest={'business_source_files':{'data/biz/'+path.name:{'bytes':path.stat().st_size,'sha256':sha(path)}}};self.assertFalse(self.proofs()[0])
 def test_duplicate_dataframe_index_does_not_remove_real_transaction(self):
  self.shop.lines.index=[0,0];clean,report=repair_shop(self.shop,self.proofs());self.assertEqual(report['removed'],1);self.assertEqual(len(clean.lines),1);self.assertEqual(clean.lines.iloc[0]['line_total'],100)

class TriageTests(unittest.TestCase):
 def classify(self,p,t,correct=True,error=None):return classify(predicted=p,target=t,as_of='2026-01-31',canonical_correct=correct,source_error=error)
 def test_identical_plan_after_executor_fix_not_training_target(self):
  self.assertEqual(self.classify(plan(),plan())['class'],'executor_repaired');self.assertFalse(self.classify(plan(),plan())['train_eligible'])
 def test_wrong_canonical_answer_not_model_target(self):
  result=self.classify(plan({'kind':'today'}),plan(),False);self.assertFalse(result['train_eligible']);self.assertEqual(result['class'],'source_scorer_or_label_discrepancy')
 def test_time_error_verified_correction(self):self.assertTrue(self.classify(plan({'kind':'today'}),plan())['train_eligible'])
 def test_equivalent_time_not_fake_correction(self):
  self.assertFalse(self.classify(plan({'kind':'this_month'}),plan())['train_eligible'])
 def test_parse_error_can_have_verified_correction(self):self.assertTrue(self.classify(None,plan())['train_eligible'])
 def test_source_failure_excluded_before_model_diff(self):self.assertFalse(self.classify(None,plan(),error='coverage_not_proven')['train_eligible'])
 def test_order_for_aggregate_not_fake_difference(self):
  p=plan();p['order']='asc';self.assertEqual(plan_difference(p,plan(),'2026-01-31'),[])
 def test_export_excludes_identical_plan_counterfactual_correction(self):
  from real_evidence import export
  from relearning import atomic
  with tempfile.TemporaryDirectory() as tmp:
   folder=Path(tmp);target=plan();q={'id':'one','as_of_date':'2026-01-31','prompt':'synthetic','business_id':'biz'};row={'id':'one','business_id':'biz','split':'learning','kind':'correction','prompt':'synthetic','response':json.dumps(target),'parent_sha256':'parent','target_verified':True}
   atomic(folder/'verified_bank.json',{'replay':[],'correction':[row]});atomic(folder/'ledger.json',{'questions':{'one':{'first_verified':False,'attempts':[{'plan':target,'state':'error'}]}}})
   bridge=SimpleNamespace(semantic=lambda q,p:True,execute=lambda q,p:{'type':'number','answer':100});policy={'parent_sha256':'parent','learning_businesses':['biz'],'calibration_businesses':['cal'],'holdout_businesses':['hold'],'min_per_kind':1,'max_per_kind':2,'min_businesses':1,**{k:k for k in ['source_manifest_sha256','split_manifest_sha256','verifier_sha256','scorer_sha256']}}
   report=export(folder,[q],bridge,lambda i:{'type':'number','answer':100},policy);self.assertEqual(report['state'],'blocked');self.assertEqual(report['excluded_non_model_corrections'],1);self.assertEqual(json.loads((folder/'PRIVATE_TARGET_EVIDENCE.json').read_text()),{})

if __name__=='__main__':unittest.main()
