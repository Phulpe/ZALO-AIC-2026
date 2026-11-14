import unittest,tempfile,json,types,sys
from pathlib import Path
from unittest.mock import patch
import post_learning_regression as target

class PostRegressionTests(unittest.TestCase):
 def test_labels_after_durable_prediction_and_no_trainable_adapter(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);source=root/'collection';source.mkdir();runtime=root/'runtime';runtime.mkdir();adapter=root/'adapter';adapter.mkdir();base=root/'base';base.mkdir();out=root/'output'
   (source/'verified_bank.json').write_text(json.dumps({'correction':[{'id':'c'}],'replay':[{'id':'r'}]}));(source/'PRIVATE_TARGET_EVIDENCE.json').write_text(json.dumps({'c':{},'r':{}}));(runtime/'RUNTIME_DATA_MANIFEST.json').write_text(json.dumps({'dataset_manifest_sha256':'data','split_manifest_sha256':'split'}));(adapter/'adapter_model.safetensors').write_bytes(b'synthetic only')
   rows=[{'id':'c','business_id':'b','kind':'correction'},{'id':'r','business_id':'b','kind':'replay'}];pool=[{'id':r['id'],'business_id':'b'} for r in rows];labels=[];loaded=[]
   def gold(qid):
    records=json.loads((out/'PRIVATE_PREDICTIONS.json').read_text());self.assertEqual(records[qid]['state'],'predicted');labels.append(qid);return True
   class Model:
    config=types.SimpleNamespace(use_cache=False)
    def eval(self):return self
   class Peft:
    @staticmethod
    def from_pretrained(model,path,**kwargs):loaded.append(kwargs);return model
   class Bridge:
    def __init__(self,*a,**kw):pass
    def predict(self,q,attempt):return {'synthetic':True}
    def semantic(self,q,plan):return True
    def execute(self,q,plan):return q['id']=='c'
   mods={'torch':types.SimpleNamespace(set_num_threads=lambda n:None,set_num_interop_threads=lambda n:None),'peft':types.SimpleNamespace(PeftModel=Peft),'modeling':types.SimpleNamespace(load_base=lambda p:(Model(),None)),'offline_guard':types.SimpleNamespace(install=lambda:None),'runtime_data':types.SimpleNamespace(load_learning_pool=lambda *a:(pool,{},gold)),'quality_gate':types.SimpleNamespace(select=lambda *a:(rows,{'dataset_sha256':'selected'})),'baseline_bridge':types.SimpleNamespace(BaselineBridge=Bridge,matches=lambda a,b:a==b),'legacy_engine':types.SimpleNamespace(prepare_shop=lambda *a,**kw:None)}
   config={'mode':'evaluation','regression_output':str(out),'verified_collection':str(source),'collection_manifest_sha256':'sealed','supervision_gate':{},'expected_training_dataset_sha256':'selected','candidate_adapter':str(adapter),'candidate_sha256':target.sha(adapter/'adapter_model.safetensors'),'runtime_root':str(runtime),'runtime_manifest_sha256':target.sha(runtime/'RUNTIME_DATA_MANIFEST.json'),'base':str(base)};request=root/'request.json';request.write_text(json.dumps(config))
   with patch.dict(sys.modules,mods),patch.object(target,'verify_export',return_value='sealed'),patch.object(Path,'symlink_to',return_value=None):
    target.main(request)
   summary=json.loads((out/'REGRESSION_SUMMARY.json').read_text());self.assertEqual(labels,['c','r']);self.assertEqual(summary['correct_to_wrong'],1);self.assertEqual(summary['correction_after_correct'],1);self.assertFalse(summary['weights_updated']);self.assertEqual(loaded[0]['is_trainable'],False)

 def test_learning_mode_rejected_before_output(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'c.json';p.write_text(json.dumps({'mode':'learning'}))
   with self.assertRaises(AssertionError):target.main(p)

if __name__=='__main__':unittest.main()
