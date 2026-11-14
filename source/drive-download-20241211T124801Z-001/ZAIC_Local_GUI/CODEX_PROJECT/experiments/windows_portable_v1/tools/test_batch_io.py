from pathlib import Path
import tempfile,json,unittest
from infer_batched import atomic_output,recover_plans,fingerprint_inputs,thinking_span,atomic_logs
class BatchIOTests(unittest.TestCase):
 def test_initial_abstain_and_order(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'submission.jsonl';qs=[{'id':'b'},{'id':'a'}]
   atomic_output(p,qs,{'a':{'id':'a','type':'number','answer':0}})
   self.assertEqual([json.loads(l) for l in p.read_text().splitlines()],[{'id':'b','type':'abstain'},{'id':'a','type':'number','answer':0}])
 def test_invalid_result_keeps_last_good_file(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'submission.jsonl';qs=[{'id':'a'}];atomic_output(p,qs,{})
   old=p.read_bytes()
   with self.assertRaises(ValueError):atomic_output(p,qs,{'a':{'id':'a','type':'number','answer':float('nan')}})
   self.assertEqual(p.read_bytes(),old)
 def test_journal_recovery_discards_only_partial_tail(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'plans.jsonl';line=json.dumps({'id':'a','plan':None})+'\n';p.write_bytes(line.encode()+b'{"id":"b"')
   self.assertEqual(set(recover_plans(p)),{'a'});self.assertEqual(p.read_text(),line)
   with p.open('a') as f:f.write(json.dumps({'id':'b','plan':None})+'\n')
   self.assertEqual(set(recover_plans(p)),{'a','b'})
 def test_conflicting_duplicate_not_silently_reused(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'plans.jsonl';p.write_text(json.dumps({'id':'a','plan':None})+'\n'+json.dumps({'id':'a','plan':{}})+'\n')
   with self.assertRaises(AssertionError):recover_plans(p)
 def test_corruption_in_complete_line_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'plans.jsonl';p.write_text('bad\n')
   with self.assertRaises(ValueError):recover_plans(p)
 def test_shortcut_and_real_root_have_identical_manifest(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'real';shop=root/'data/biz';shop.mkdir(parents=True);(shop/'README.md').write_text('schema')
   alias=Path(d)/'shortcut';(__import__('_winapi').CreateJunction(str(root),str(alias)) if __import__('os').name=='nt' else alias.symlink_to(root,target_is_directory=True))
   self.assertEqual(fingerprint_inputs(alias,{'biz'}),fingerprint_inputs(root,{'biz'}))
 def test_manifest_tracks_content_and_never_reference_answers(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);shop=root/'data/biz';shop.mkdir(parents=True);f=shop/'s.csv';f.write_text('A,1')
   (root/'answers.jsonl').write_text('REFERENCE_LABELS')
   _,old=fingerprint_inputs(root,{'biz'});f.write_text('A,2');_,new=fingerprint_inputs(root,{'biz'})
   self.assertNotEqual(old['data/biz/s.csv']['sha256'],new['data/biz/s.csv']['sha256']);self.assertNotIn('answers.jsonl',new)
 def test_exact_untruncated_generated_thinking_and_log_contract(self):
  thought='line one\n'+'x'*10000
  self.assertEqual(thinking_span('<think>'+thought+'</think><eos>','<eos>'),thought)
  self.assertIsNone(thinking_span('{"metric":"revenue"}'))
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'logs/q.jsonl';r={'id':'q','answer':{'type':'abstain'},'elapsed_ms':0,'thinking':thought,'code':'planner.py::answer','inputs':{},'output':{'type':'abstain'}}
   atomic_logs(p,[{'id':'q'}],{'q':r});self.assertEqual(json.loads(p.read_text())['thinking'],thought)
if __name__=='__main__':unittest.main()
