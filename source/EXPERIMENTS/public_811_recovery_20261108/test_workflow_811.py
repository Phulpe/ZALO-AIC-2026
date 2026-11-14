import unittest,copy,ast
from pathlib import Path
from workflow_811 import validate_request,validate_public_result,PARENT_SHA

class WorkflowTests(unittest.TestCase):
    def inputs(self):
        return ({'training_objective':'verified_replay_refresh','budget_approved':True,'max_steps':4,'total_gpu_seconds':3600,'parent_sha256':PARENT_SHA},[{'id':str(i),'split':'learning','target_verified':True,'kind':'replay','parent_sha256':PARENT_SHA,'business_id':'synthetic'} for i in range(10)],{'parent_sha256':PARENT_SHA,'learning_businesses':['synthetic']})
    def test_explicit_real_replay_pilot_accepted(self):validate_request(*self.inputs())
    def test_correction_objective_not_silently_relaxed(self):
        c,r,p=self.inputs();c['training_objective']='correction_replay'
        with self.assertRaises(ValueError):validate_request(c,r,p)
    def test_budget_and_parent_pinned(self):
        for key,value in [('budget_approved',False),('max_steps',5),('total_gpu_seconds',7200),('parent_sha256','smoke')]:
            c,r,p=self.inputs();c[key]=value
            with self.assertRaises(ValueError):validate_request(c,r,p)
    def test_public_split_unverified_and_wrong_parent_rejected(self):
        for key,value in [('split','public_test'),('target_verified',False),('parent_sha256','other')]:
            c,r,p=self.inputs();r[0][key]=value
            with self.assertRaises(ValueError):validate_request(c,r,p)
    def test_duplicate_and_missing_targets_rejected(self):
        c,r,p=self.inputs();r[-1]['id']='0'
        with self.assertRaises(ValueError):validate_request(c,r,p)
        with self.assertRaises(ValueError):validate_request(c,r[:-1],p)
    def test_public_generation_arguments_unchanged(self):
        def calls(path):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            return [ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='generate']
        root=Path(__file__).parent
        self.assertEqual(calls(root/'public_infer_811.py'),calls(root/'baseline/tools/infer_batched.py'))
    def test_submission_not_completed_by_placeholder_count(self):
        qs=[{'id':str(i)} for i in range(1000)];predictions=[{'id':q['id'],'type':'abstain'} for q in qs]
        with self.assertRaises(ValueError):validate_public_result({'state':'running','n':1000,'question_logs':0,'network_attempts':0},qs,predictions)
        validate_public_result({'state':'completed','n':1000,'question_logs':1000,'network_attempts':0},qs,predictions)

