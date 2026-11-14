import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
from relearning import sha,atomic
from learning_probe import collect
from baseline_bridge import BaselineBridge,canonical_total
from planner import context,prompt_text
from test_core import fixture

class ProbeTests(unittest.TestCase):
    def test_budget_rejected_before_io(self):
        with self.assertRaises(ValueError):collect(None,None,{'learning_probe_max_questions':100})
    def test_blind_probe_builds_verified_correction_and_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp);root=temp/'runtime';folder=root/'data/s';folder.mkdir(parents=True);(folder/'README.md').write_text('synthetic')
            atomic(root/'RUNTIME_DATA_MANIFEST.json',{'dataset_manifest_sha256':'data','split_manifest_sha256':'split','files':{'data/s/README.md':{'sha256':sha(folder/'README.md')}}})
            shop=fixture();pool=[]
            for key,text in [('a','<REDACTED_COMPETITION_QUESTION>'),('b','Tổng số lượng bán tháng 1 năm 2026 là bao nhiêu?')]:
                pool.append({'id':key,'business_id':'s','split':'learning','source_ref':'s','as_of_date':'2026-01-31','prompt':prompt_text(context(shop,text,'2026-01-31',''))})
            split={'groups':{'learning':['s'],'calibration':['c'],'audit_exposed':['h']},'pilot_ids':['a','b']}
            class FakeBridge(BaselineBridge):
                def predict(self,q,attempt):
                    p=canonical_total(q)
                    if q['id']=='b':p['metric']='revenue'
                    return p
            class Model:
                training=True
                def train(self,state):self.training=state
            def gold(key):return {'type':'number','answer':300 if key=='a' else 12}
            cfg={'learning_probe_max_questions':2,'runtime_root':str(root),'runtime_manifest_sha256':sha(root/'RUNTIME_DATA_MANIFEST.json'),'output_root':str(temp/'models'),'parent_sha256':'parent'}
            with patch('learning_probe.load_learning_pool',return_value=(pool,split,gold)),patch('baseline_bridge.BaselineBridge',FakeBridge),patch('legacy_engine.prepare_shop',return_value=shop):
                result=collect(Model(),None,cfg)
            self.assertTrue(result['verified_dataset_ready']);self.assertEqual(result['replay'],1);self.assertEqual(result['correction'],1);self.assertFalse(result['optimizer_executed']);self.assertEqual(result['full_pool_coverage']['attempt_coverage'],1)
