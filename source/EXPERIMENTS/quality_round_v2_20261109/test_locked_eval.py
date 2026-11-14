import unittest,tempfile,json,sys,types
from pathlib import Path
from unittest.mock import patch
from locked_eval import evaluate
from relearning import sha
class EvalTests(unittest.TestCase):
 def test_gold_hash_checked_after_predictions(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);runtime=p/'runtime';(runtime/'calibration').mkdir(parents=True);(runtime/'data').mkdir();(runtime/'rulebook.md').write_text('synthetic rules')
   (runtime/'calibration/questions.jsonl').write_text(json.dumps({'id':'synthetic','business_id':'a','question':'synthetic','as_of_date':'2026-01-01'})+'\n')
   (runtime/'calibration/gold.jsonl').write_text(json.dumps({'id':'synthetic','type':'number','answer':3})+'\n');adapter=p/'adapter';adapter.mkdir();base=p/'base';base.mkdir()
   output=p/'eval';observed=[]
   def fake_main():
    observed.append('blind_inference');q=Path(sys.argv[sys.argv.index('--output-dir')+1]);q.mkdir();(q/'submission.jsonl').write_text(json.dumps({'id':'synthetic','type':'number','answer':3})+'\n');(q/'runtime.json').write_text(json.dumps({'state':'completed','question_logs':1,'network_attempts':0}))
   c={'runtime_root':str(runtime),'base':str(base),'calibration_questions_sha256':sha(runtime/'calibration/questions.jsonl'),'calibration_gold_sha256':'wrong','calibration_manifest_sha256':'frozen'}
   with patch.dict(sys.modules,{'public_infer_811':types.SimpleNamespace(main=fake_main)}):
    with self.assertRaises(AssertionError):evaluate(c,adapter,output)
   self.assertEqual(observed,['blind_inference']);self.assertTrue((output/'predictions/submission.jsonl').exists());self.assertFalse((output/'EVALUATION.json').exists())
