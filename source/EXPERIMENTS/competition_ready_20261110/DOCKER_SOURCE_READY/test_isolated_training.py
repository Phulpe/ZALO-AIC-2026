import unittest,tempfile,sys,json,os
from pathlib import Path
from unittest.mock import patch
from isolated_training import run_stage,train_verified_isolated,ChildStageError

class ProcessBoundaryTests(unittest.TestCase):
 def setUp(self):self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
 def run_child(self,text,timeout=5):return run_stage([sys.executable,'-X','utf8','-c',text],self.root/'stage',timeout)
 def test_child_really_exits_before_parent_evaluation(self):
  marker=self.root/'child.json';text='import json,os;from pathlib import Path;Path('+repr(str(marker))+').write_text(json.dumps({"pid":os.getpid()}))'
  result=self.run_child(text);self.assertEqual(result['state'],'completed');self.assertNotEqual(json.loads(marker.read_text())['pid'],os.getpid())
  self.assertEqual(json.loads((self.root/'stage/STAGE_RESULT.json').read_text())['returncode'],0)
 def test_failed_child_preserves_checkpoint_and_does_not_continue(self):
  ck=self.root/'checkpoint';after=[]
  with self.assertRaises(ChildStageError):
   self.run_child('from pathlib import Path;Path('+repr(str(ck))+').write_text("saved");raise RuntimeError("synthetic failure")');after.append('evaluation')
  self.assertEqual(ck.read_text(),'saved');self.assertEqual(after,[]);self.assertEqual(json.loads((self.root/'stage/STAGE_RESULT.json').read_text())['state'],'failed')
 def test_timeout_kills_and_waits_without_removing_checkpoint(self):
  ck=self.root/'checkpoint'
  with self.assertRaises(ChildStageError):self.run_child('import time;from pathlib import Path;Path('+repr(str(ck))+').write_text("saved");time.sleep(15)',1)
  self.assertEqual(ck.read_text(),'saved');result=json.loads((self.root/'stage/STAGE_RESULT.json').read_text());self.assertTrue(result['child_waited_after_kill'])
 def test_private_error_text_not_exposed_by_parent_exception(self):
  with self.assertRaises(ChildStageError) as error:self.run_child('raise RuntimeError("PRIVATE_SYNTHETIC_SENTINEL")')
  self.assertNotIn('PRIVATE_SYNTHETIC_SENTINEL',str(error.exception));self.assertIn('PRIVATE_SYNTHETIC_SENTINEL',(self.root/'stage/PRIVATE_CHILD.log').read_text())
 def test_stage_never_overwrites_existing_outputs(self):
  self.run_child('pass')
  with self.assertRaises(FileExistsError):self.run_child('raise RuntimeError()')
 def test_nonfinite_or_missing_timeout_rejected(self):
  for value in [True,0,-1,float('inf'),float('nan'),None]:
   with self.assertRaises(ValueError):run_stage([sys.executable,'-c','pass'],self.root/'other',value)
 def test_fresh_and_resume_never_become_warm_start(self):
  for mode in ['fresh','resume',None]:
   with patch('isolated_training.run_stage') as run:
    with self.assertRaises(ValueError):train_verified_isolated({'mode':'learning','init_mode':mode},[],{})
    run.assert_not_called()
 def test_missing_budget_blocks_before_child(self):
  with patch('isolated_training.run_stage') as run:
   with self.assertRaises(ValueError):train_verified_isolated({'mode':'learning','init_mode':'warm_start','collect_learning_probe':False},[],{})
   run.assert_not_called()
 def test_protected_output_blocks_before_writes(self):
  base=self.root/'base';base.mkdir();adapter=self.root/'adapter';adapter.mkdir();config={'mode':'learning','init_mode':'warm_start','user_authorized':True,'budget_approved':True,'max_seconds':5,'child_timeout_seconds':10,'base':str(base),'adapter':str(adapter),'output_root':str(base/'frozen/model_runs')}
  with self.assertRaisesRegex(ValueError,'protected assets'):train_verified_isolated(config,[],{})
  self.assertFalse((base/'frozen').exists())

if __name__=='__main__':unittest.main()
