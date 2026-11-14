import unittest,tempfile,json,types,sys,contextlib
from pathlib import Path
from unittest.mock import Mock,patch
from relearning import LearningRound
from prediction_errors import BlindPredictionError

class PredictionTests(unittest.TestCase):
    def test_invalid_blind_output_saved_before_gold_and_corrected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);events=[]
            def predict(q,n):events.append('predict');raise BlindPredictionError('synthetic malformed response')
            def gold(key):
                stored=json.loads((path/'ledger.json').read_text())['questions']['fake']['attempts']
                self.assertEqual(len(stored),2)
                self.assertTrue(all(a['prediction_text']=='synthetic malformed response' for a in stored))
                events.append('gold');return 1
            config={'mode':'learning','attempts_per_question':2,'learning_businesses':['a'],'calibration_businesses':['b'],'holdout_businesses':['c'],'parent_sha256':'parent'}
            pool=[{'id':'fake','split':'learning','business_id':'a','prompt':'synthetic'}]
            loop=LearningRound(path,config,pool,gold,predict,lambda q,p:p,lambda p:None,lambda q,p:True,lambda a,b:a==b,lambda q,row,g:1)
            metrics,replay,correction=loop.process()
            self.assertEqual(events,['predict','predict','gold'])
            self.assertEqual(len(correction),1);self.assertFalse(replay)
            self.assertEqual(metrics['verified_correction_coverage'],1)
    def test_no_output_timeout_does_not_open_gold(self):
        with tempfile.TemporaryDirectory() as temp:
            gold=Mock(side_effect=AssertionError('Gold must stay closed'))
            config={'mode':'learning','attempts_per_question':1,'learning_businesses':['a'],'calibration_businesses':['b'],'holdout_businesses':['c'],'parent_sha256':'p'}
            pool=[{'id':'fake','split':'learning','business_id':'a','prompt':'synthetic'}]
            loop=LearningRound(temp,config,pool,gold,Mock(side_effect=TimeoutError()),lambda q,p:p,lambda p:None,lambda q,p:True,lambda a,b:a==b)
            metrics,_,_=loop.process();self.assertEqual(metrics['unresolved'],1);gold.assert_not_called()
    def test_exception_string_never_discloses_response(self):
        exc=BlindPredictionError('synthetic private response')
        self.assertNotIn('synthetic private response',str(exc))
    def test_generation_cache_and_malformed_response_capture(self):
        from baseline_bridge import BaselineBridge
        class Inputs(dict):
            def to(self,device):return self
        class Generated:
            def __getitem__(self,key):return [42]
        class Tokenizer:
            pad_token_id=0;eos_token_id=1
            def apply_chat_template(self,*a,**k):return 'synthetic'
            def __call__(self,*a,**k):return Inputs(input_ids=types.SimpleNamespace(shape=[1,2]))
            def decode(self,*a,**k):return 'synthetic malformed response'
        model=Mock();model.generate.return_value=Generated()
        fake_torch=types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        with patch.dict(sys.modules,{'torch':fake_torch}):
            with self.assertRaises(BlindPredictionError) as caught:BaselineBridge(model,Tokenizer(),{}).predict({'prompt':'synthetic'},1)
        self.assertEqual(caught.exception.prediction_text,'synthetic malformed response')
        self.assertIs(model.generate.call_args.kwargs['use_cache'],True)
