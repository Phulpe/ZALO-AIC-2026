import unittest,json,math,importlib.util
from time_objective import time_span,token_focus,payload_spans,payload_focus,validate_weight,weighted_token_loss

class SpanTests(unittest.TestCase):
 def test_top_level_not_nested_or_quoted(self):
  s=' {"note":"escaped \\"time\\"", "nested":{"time":1}, "time" : {"kind":"yesterday"} }'
  # Use JSON encoding so escaped strings cannot be mistaken for object keys.
  s=json.dumps({'note':'"time"','nested':{'time':1},'time':{'kind':'yesterday'}})
  a,b=time_span(s);self.assertEqual(json.loads(s[a:b]),{'kind':'yesterday'})
 def test_duplicate_rejected(self):
  for s in ['{"time":{},"time":{}}','{"time":{"kind":"today","kind":"yesterday"}}']:
   with self.assertRaises(ValueError):time_span(s)
 def test_missing_time_rejected(self):
  for s in ['{}','[]','{"time":null}']:
   with self.assertRaises(ValueError):time_span(s)
 def test_offsets_overlap_eos_excluded(self):
  self.assertEqual(token_focus([(0,2),(2,5),(5,9),(9,12),(0,0)],(4,9),12),[False,True,True,False,False])
 def test_no_time_tokens_rejected(self):
  with self.assertRaises(ValueError):token_focus([(0,1)],(4,9),12)
 def test_weight_finite_and_bounded(self):
  self.assertEqual(validate_weight(4),4.)
  for v in [True,0,9,float('nan'),float('inf'),'4']:
   with self.assertRaises(ValueError):validate_weight(v)
 def test_payload_excludes_fixed_keys_and_punctuation(self):
  s='{"metric":"revenue","time":{"kind":"yesterday"}}'
  spans=payload_spans(s);self.assertEqual([s[a:b] for a,b in spans],['yesterday'])
  offsets=[(i,i+1) for i in range(len(s))];mask=payload_focus(offsets,s);self.assertEqual(''.join(c for c,v in zip(s,mask) if v),'yesterday')
 def test_range_payload_and_escaped_strings(self):
  s=json.dumps({'time':{'kind':'range','start':'2024-02-01','end':'2024-02-29'}})
  self.assertEqual([s[a:b] for a,b in payload_spans(s)],['range','2024-02-01','2024-02-29'])

@unittest.skipUnless(importlib.util.find_spec('torch'),'torch CPU is supplied by Kaggle; no local install needed')
class LossTests(unittest.TestCase):
 def test_weighted_mean_and_gradient(self):
  import torch
  x=torch.tensor([[[0.,0.],[2.,0.]]],requires_grad=True);y=torch.tensor([[0,1]]);w=torch.tensor([[1.,4.]]);f=torch.tensor([[False,True]])
  loss,stats=weighted_token_loss(x,y,w,f)
  expected=(math.log(2)+4*math.log(1+math.exp(2)))/5
  self.assertAlmostEqual(loss.item(),expected,places=6);loss.backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertEqual(stats['time_tokens'],1);self.assertEqual(stats['time_correct_tokens'],0)
  self.assertAlmostEqual(x.grad[0,0,0].item(),-.5/5,places=6);self.assertAlmostEqual(x.grad[0,1,1].item(),(1/(1+math.exp(2))-1)*4/5,places=6)
 def test_weight_one_matches_original_ce(self):
  import torch
  import torch.nn.functional as F
  x=torch.tensor([[[1.,2.,3.],[3.,1.,2.]]]);y=torch.tensor([[1,0]]);loss,_=weighted_token_loss(x,y,torch.ones_like(y,dtype=torch.float32),torch.tensor([[True,False]]));self.assertAlmostEqual(loss.item(),F.cross_entropy(x.reshape(-1,3),y.reshape(-1)).item(),places=6)
 def test_mask_and_shape_fail_closed(self):
  import torch
  x=torch.zeros(1,2,3);y=torch.tensor([[0,-100]])
  with self.assertRaises(ValueError):weighted_token_loss(x,y,torch.ones(1,2),torch.ones(1,2,dtype=torch.bool))
  with self.assertRaises(ValueError):weighted_token_loss(x,y[:,:1],torch.ones(1,2),torch.ones(1,2,dtype=torch.bool))

if __name__=='__main__':unittest.main()
