import unittest,tempfile,json
from pathlib import Path
from unittest.mock import Mock,patch
import relearning as r

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)
        self.cfg={'mode':'learning','attempts_per_question':3,'learning_businesses':['a'],'calibration_businesses':['b'],'holdout_businesses':['c'],'parent_sha256':'parent'}
        self.pool=[{'id':str(i),'split':'learning','business_id':'a','prompt':'synthetic','as_of_date':'2026-01-01'} for i in range(3)]
    def round(self,predict=lambda q,n:1,execute=lambda q,p:p,semantic=lambda q,p:True,correct=None,gold=None,pool=None,cfg=None):
        return r.LearningRound(self.p,cfg or self.cfg,self.pool if pool is None else pool,gold or (lambda i:1),predict,execute,lambda p:None,semantic,lambda a,b:a==b,correct)
    def test_coverage(self):
        m,replay,c=self.round().process();self.assertEqual(m['attempt_coverage'],1);self.assertEqual(len(replay),3)
    def test_gold_order(self):
        events=[]
        def gold(i):
            ledger=json.loads((self.p/'ledger.json').read_text());self.assertEqual(ledger['questions'][i]['attempts'][0]['state'],'predicted');events.append('gold');return 1
        def predict(q,n):self.assertNotIn('gold',q);events.append('predict');return 1
        self.round(predict=predict,gold=gold).process();self.assertEqual(events,['predict','gold']*3)
    def test_gold_field_rejected(self):
        self.pool[0]['gold']=1
        with self.assertRaises(ValueError):self.round()
    def test_wrong_split(self):
        self.pool[0]['split']='holdout'
        with self.assertRaises(ValueError):self.round()
    def test_overlap_split(self):
        self.cfg['holdout_businesses']=['a']
        with self.assertRaises(ValueError):self.round()
    def test_duplicate(self):
        self.pool[1]['id']='0'
        with self.assertRaises(ValueError):self.round()
    def test_retry_bounded(self):
        m,rep,c=self.round(predict=lambda q,n:0).process();l=json.loads((self.p/'ledger.json').read_text());self.assertEqual(sum(len(x['attempts']) for x in l['questions'].values()),9);self.assertEqual(m['unresolved'],3);self.assertFalse(rep or c)
    def test_invalid_retry(self):
        self.cfg['attempts_per_question']=100
        with self.assertRaises(ValueError):self.round()
    def test_error_retained(self):
        def fail(q,n):raise TimeoutError()
        m,_,_=self.round(predict=fail).process();self.assertEqual(m['attempt_coverage'],1);self.assertEqual(m['unresolved'],3)
    def test_executor_error(self):
        def fail(q,p):raise ValueError()
        self.round(execute=fail).process();l=json.loads((self.p/'ledger.json').read_text());self.assertEqual(l['questions']['0']['attempts'][0]['error_class'],'executor')
    def test_correction_verified(self):
        m,_,c=self.round(predict=lambda q,n:0,correct=lambda q,row,g:1).process();self.assertEqual(len(c),3);self.assertEqual(m['verified_correction_coverage'],1)
    def test_bad_correction_not_target(self):
        _,_,c=self.round(predict=lambda q,n:0,correct=lambda q,row,g:0).process();self.assertEqual(c,[])
    def test_accidental_match_no_replay(self):
        _,rep,c=self.round(semantic=lambda q,p:False,correct=lambda q,row,g:1).process();self.assertEqual(rep+c,[])
    def test_semantic_mismatch_is_model_error_not_ingestion(self):
        execute=Mock()
        self.round(semantic=lambda q,p:False,execute=execute).process()
        row=json.loads((self.p/'ledger.json').read_text())['questions']['0']
        self.assertEqual(row['attempts'][0]['error_class'],'model');execute.assert_not_called()
    def test_evaluation_not_learning(self):
        self.cfg['mode']='evaluation'
        with self.assertRaises(ValueError):self.round()
    def test_no_ledger_overwrite(self):
        self.round()
        with self.assertRaises(FileExistsError):self.round()
    def test_output_protection(self):
        with self.assertRaises(ValueError):r.new_run(self.p/'baseline/out',{},[self.p/'baseline'])
    def test_warm_no_fallback(self):
        loader=Mock(side_effect=RuntimeError('load failed'))
        with patch.object(r,'verify_parent',return_value='parent'):
            with self.assertRaises(RuntimeError):r.warm_start(object(),self.p,{'init_mode':'warm_start'},loader)
        self.assertEqual(loader.call_count,1);self.assertTrue(loader.call_args.kwargs['is_trainable'])
    def test_resume_rejected(self):
        with self.assertRaises(ValueError):r.warm_start(object(),self.p,{'init_mode':'resume'},Mock())
    def test_wrong_parent(self):
        (self.p/'adapter_model.safetensors').write_bytes(b'fake')
        with self.assertRaises(ValueError):r.verify_parent(self.p)
    def test_regression_missing(self):
        with self.assertRaises(ValueError):r.regressions({'a':True},{})
    def test_regression(self):self.assertEqual(r.regressions({'a':True,'b':False},{'a':False,'b':True}),['a'])
    def test_export_corruption(self):
        (self.p/'a').write_text('before');r.seal(self.p);(self.p/'a').write_text('after')
        with self.assertRaises(ValueError):r.verify_export(self.p)
    def test_best_not_overwritten(self):
        run=self.p/'run';run.mkdir();(run/'a').write_text('ok');r.seal(run)
        state=self.p/'STATE.json';r.atomic(state,{'BEST_MODEL':'baseline','LATEST_CHECKPOINT':None,'LATEST_SYNCED':None})
        self.assertFalse(r.track(state,run,'promote',{}));self.assertEqual(json.loads(state.read_text())['BEST_MODEL'],'baseline')
    def test_sync_requires_hash(self):
        run=self.p/'run';run.mkdir();(run/'a').write_text('ok');r.seal(run)
        with self.assertRaises(ValueError):r.track(self.p/'STATE.json',run,'sync',{'saved_version_url':'saved','remote_manifest_sha256':'bad'})
    def test_unverified_mixture(self):
        with self.assertRaises(ValueError):r.mixture([{'id':'a','target_verified':False,'split':'learning','business_id':'a'}],[],['a'])
    def test_balanced_mixture(self):
        def row(i):return {'id':i,'target_verified':True,'split':'learning','business_id':'a'}
        self.assertEqual(len(r.mixture([row('a'),row('b')],[row('c')],['a'])),2)
    def test_checkpoint_pointer_separate(self):
        run=self.p/'run';ck=run/'checkpoint';ck.mkdir(parents=True)
        for n in ['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json']:(ck/n).write_text('synthetic')
        r.seal(run);path=self.p/'STATE.json';r.track(path,run,'checkpoint',{})
        state=json.loads(path.read_text());self.assertIsNotNone(state['LATEST_CHECKPOINT']);self.assertIsNone(state['BEST_MODEL']);self.assertIsNone(state['LATEST_SYNCED'])
    def test_mismatched_recovery_rejected(self):
        self.round();cfg={**self.cfg,'source':'changed'}
        with self.assertRaises(ValueError):r.LearningRound(self.p,cfg,self.pool,lambda i:1,lambda q,n:1,lambda q,p:p,lambda p:None,lambda q,p:True,lambda a,b:a==b,recover=True)
    def test_completed_recovery_no_retries(self):
        self.round().process();predict=Mock(side_effect=RuntimeError())
        obj=r.LearningRound(self.p,self.cfg,self.pool,lambda i:1,predict,lambda q,p:p,lambda p:None,lambda q,p:True,lambda a,b:a==b,recover=True)
        metrics,replay,_=obj.process();self.assertEqual(metrics['attempt_coverage'],1);self.assertEqual(len(replay),3);predict.assert_not_called()
    def test_trainable_warm_start(self):
        model=Mock();param=Mock();param.requires_grad=True;model.parameters.return_value=[param]
        loader=Mock(return_value=model)
        with patch.object(r,'verify_parent',return_value='parent'):
            actual,meta=r.warm_start(object(),self.p,{'init_mode':'warm_start'},loader)
        self.assertIs(actual,model);self.assertEqual(meta['optimizer_init'],'new');self.assertEqual(meta['scheduler_init'],'new')
    def test_untrainable_adapter_rejected(self):
        model=Mock();model.parameters.return_value=[]
        with patch.object(r,'verify_parent',return_value='parent'):
            with self.assertRaises(ValueError):r.warm_start(object(),self.p,{'init_mode':'warm_start'},Mock(return_value=model))
    def test_partial_attempt_recovery_does_not_claim_coverage(self):
        self.round();p=self.p/'ledger.json';obj=json.loads(p.read_text());obj['questions']['0']['attempts']=[{'state':'started'}];r.atomic(p,obj)
        with self.assertRaises(ValueError):r.LearningRound(self.p,self.cfg,self.pool,lambda i:1,lambda q,n:1,lambda q,p:p,lambda p:None,lambda q,p:True,lambda a,b:a==b,recover=True)

if __name__=='__main__':unittest.main()
