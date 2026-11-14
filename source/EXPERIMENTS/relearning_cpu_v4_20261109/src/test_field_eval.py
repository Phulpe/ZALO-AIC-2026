import unittest,tempfile,types,json,sys,importlib.util
from pathlib import Path
from unittest.mock import patch
from field_eval import evaluate_rows,verify_loaded
from pilot_gate import learning_gate

class BlindEvalTests(unittest.TestCase):
 def run_fake(self,fail=False):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/'out';gold_calls=[]
   class Bridge:
    def __init__(self,*a,**kw):pass
    def predict(self,q,attempt):
     if fail:raise MemoryError('synthetic infrastructure error')
     return {'synthetic':True}
    def semantic(self,*a):return True
    def execute(self,*a):return 1
   def gold(i):
    self.assertEqual(json.loads((out/'PRIVATE_PREDICTIONS.json').read_text())[i]['state'],'predicted');gold_calls.append(i);return 1
   rows=[{'id':'c','business_id':'b','kind':'correction'},{'id':'r','business_id':'b','kind':'replay'}]
   with patch.dict(sys.modules,{'baseline_bridge':types.SimpleNamespace(BaselineBridge=Bridge,matches=lambda a,b:a==b)}):report=evaluate_rows(None,None,rows,rows,gold,None,out)
   return report,gold_calls
 def test_prediction_persisted_before_gold(self):
  report,called=self.run_fake();self.assertEqual(called,['c','r']);self.assertTrue(learning_gate(report)['passed'])
 def test_infrastructure_error_not_fake_output_or_gold(self):
  report,called=self.run_fake(True);self.assertEqual(called,[]);self.assertEqual(report['real_output_ids'],0)
  with self.assertRaises(ValueError):learning_gate(report)

@unittest.skipUnless(importlib.util.find_spec('torch'),'torch CPU supplied by Kaggle')
class StrictLoadTests(unittest.TestCase):
 def test_loaded_weights_must_equal_saved_weights(self):
  import torch
  from safetensors.torch import save_file
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);weights={'synthetic.lora_A.weight':torch.tensor([[1.,2.]])};save_file(weights,str(p/'adapter_model.safetensors'))
   with patch('peft.get_peft_model_state_dict',return_value=weights):self.assertTrue(verify_loaded(None,p)['all_tensor_values_match'])
   for bad in [{},{'synthetic.lora_A.weight':torch.tensor([[1.,3.]])}]:
    with patch('peft.get_peft_model_state_dict',return_value=bad),self.assertRaises(ValueError):verify_loaded(None,p)

if __name__=='__main__':unittest.main()
