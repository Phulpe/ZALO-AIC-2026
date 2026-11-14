import unittest,json,copy
from quality_gate import select,digest

def fixture():
 config={'parent_sha256':'p','learning_businesses':['a','b','c','d','e'],'calibration_businesses':['cal'],'holdout_businesses':['hold'],'max_per_kind':5,'min_per_kind':5,'min_businesses':2,'selection_policy':'time_business_metric_v1','min_replay_time_buckets':5,**{k:k for k in ['source_manifest_sha256','split_manifest_sha256','verifier_sha256','scorer_sha256']}}
 rows=[];evidence={}
 periods=['month']*20+['today','yesterday','this_week','last_week']
 for kind,values in [('correction',['yesterday']*5),('replay',periods)]:
  for i,period in enumerate(values):
   key=kind+str(i);plan={'metric':'revenue','time':{'kind':period},'filters':[],'group_by':None,'order':'desc','limit':None,'unit':None,'answer_type':'number','unresolved':[]};prompt='synthetic '+key;row={'id':key,'business_id':config['learning_businesses'][i%5],'split':'learning','kind':kind,'parent_sha256':'p','target_verified':True,'prompt':prompt,'response':json.dumps(plan)};rows.append(row)
   ev={k:config[k] for k in ['source_manifest_sha256','split_manifest_sha256','verifier_sha256','scorer_sha256']};ev.update({k:True for k in ['schema','semantic','execution','scorer','blind_prediction_persisted','gold_after_prediction']});ev.update(parent_sha256='p',plan_sha256=digest(plan),prompt_sha256=digest(prompt),first_pass_verified_correct=kind=='replay',temporal_bucket=period);evidence[key]=ev
 return rows,evidence,config

class TemporalSelectionTests(unittest.TestCase):
 def test_rare_periods_survive_dominant_month_rows(self):
  rows,ev,c=fixture();out,r=select(rows,ev,c);self.assertEqual(len(out),10);self.assertEqual(set(r['replay_time_buckets']),{'month','today','yesterday','this_week','last_week'});self.assertEqual(r['correction_time_buckets'],{'yesterday':5});self.assertEqual(len({x['id'] for x in out}),10)
 def test_selection_is_deterministic_without_input_order_dependence(self):
  rows,ev,c=fixture();self.assertEqual(select(rows,ev,c),select(list(reversed(rows)),ev,c))
 def test_missing_bucket_never_silently_uses_legacy_selection(self):
  rows,ev,c=fixture();del ev[rows[0]['id']]['temporal_bucket']
  with self.assertRaisesRegex(ValueError,'temporal evidence'):select(rows,ev,c)
 def test_unknown_bucket_blocked(self):
  rows,ev,c=fixture();ev[rows[0]['id']]['temporal_bucket']='inferred_from_gold'
  with self.assertRaises(ValueError):select(rows,ev,c)
 def test_insufficient_available_periods_blocks_new_gate(self):
  rows,ev,c=fixture();c['min_replay_time_buckets']=6
  with self.assertRaisesRegex(ValueError,'temporal replay diversity'):select(rows,ev,c)
 def test_no_new_labels_or_oversampling_and_business_diversity(self):
  rows,ev,c=fixture();before=copy.deepcopy(rows);out,r=select(rows,ev,c);self.assertEqual(rows,before);self.assertTrue(all(x in rows for x in out));self.assertGreaterEqual(r['unique_businesses'],2)

if __name__=='__main__':unittest.main()
