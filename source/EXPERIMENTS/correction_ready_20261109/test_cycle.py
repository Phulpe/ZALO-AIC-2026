import tempfile,unittest,json
from pathlib import Path
from cycle import run_cycle
from relearning import atomic,seal

class CycleTests(unittest.TestCase):
    def test_end_to_end_regression_preserves_best(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);atomic(root/'STATE.json',{'BEST_MODEL':'parent','LATEST_SYNCED':None,'LATEST_CHECKPOINT':None})
            cfg={'output_root':str(root),'mode':'learning','init_mode':'warm_start','parent_sha256':'parent','attempts_per_question':1,'learning_businesses':['s'],'calibration_businesses':['c'],'holdout_businesses':['h'],'minimum_improvement':0,'max_regressions':0}
            pool=[{'id':str(i),'split':'learning','business_id':'s','prompt':'synthetic'} for i in range(2)]
            seen=[]
            def learner(config,rows):
                seen.extend(rows);p=root/'candidate';p.mkdir();(p/'adapter').write_text('synthetic');seal(p);return p
            def evaluate(model,mode):
                self.assertEqual(mode,'evaluation')
                return {'calibration_manifest_sha256':'fixed','verdicts':{'cal':model=='parent'},'metric':1 if model=='parent' else 2,'gates_passed':True}
            run=run_cycle(cfg,pool,lambda i:1,lambda q,n:1 if q['id']=='0' else 0,lambda q,p:p,lambda p:None,lambda q,p:True,lambda a,b:a==b,lambda q,row,g:1,evaluate,learner)
            result=json.loads((run/'CYCLE_RESULT.json').read_text());self.assertFalse(result['promoted']);self.assertEqual(result['correct_to_wrong'],['cal']);self.assertEqual(len(seen),2);self.assertEqual(json.loads((root/'STATE.json').read_text())['BEST_MODEL'],'parent')
    def test_permission_gate_before_any_processing(self):
        with self.assertRaises(ValueError):run_cycle({'mode':'learning','init_mode':'warm_start','competition_assets':True},[],None,None,None,None,None,None,None,None,None)
