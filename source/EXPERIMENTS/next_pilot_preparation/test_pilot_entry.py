import unittest
from pilot_entry import PARENT,REVISION,validate_config,validate_rows,run_pilot

class PilotGateTests(unittest.TestCase):
    def config(self):return dict(budget_approved=True,gpu_confirmed=True,mode='learning',init_mode='warm_start',parent_sha256=PARENT,revision=REVISION,max_steps=4,max_seconds=900,lr=5e-6,learning_businesses=['synthetic'])
    def rows(self):return [dict(id=str(i),kind=k,target_verified=True,split='learning',business_id='synthetic',parent_sha256=PARENT) for i,k in enumerate(['replay','correction'])]
    def test_default_budget_blocks_before_filesystem(self):
        c=self.config();c['budget_approved']=False
        with self.assertRaises(ValueError):run_pilot(c)
    def test_fresh_and_resume_rejected(self):
        for mode in ['fresh','resume']:
            c=self.config();c['init_mode']=mode
            with self.assertRaises(ValueError):validate_config(c)
    def test_synthetic_candidate_cannot_be_parent(self):
        c=self.config();c['parent_sha256']='e71bca7cadce97638cb6f317cc87f15d5cbced580c8f9a4be3e764320c03895d'
        with self.assertRaises(ValueError):validate_config(c)
    def test_valid_budget_and_rows(self):validate_config(self.config());validate_rows(self.config(),self.rows())
    def test_excess_budget_rejected(self):
        for field,value in [('max_steps',5),('max_seconds',901),('lr',1e-4)]:
            c=self.config();c[field]=value
            with self.assertRaises(ValueError):validate_config(c)
    def test_replay_only_cannot_start_training(self):
        with self.assertRaises(ValueError):validate_rows(self.config(),self.rows()[:1])
    def test_wrong_split_unverified_and_duplicates_rejected(self):
        for field,value in [('split','calibration'),('target_verified',False),('id','0')]:
            rows=self.rows();rows[1][field]=value
            with self.assertRaises(ValueError):validate_rows(self.config(),rows)
    def test_missing_dataset_blocks_before_model_import(self):
        with self.assertRaises(ValueError):run_pilot(self.config())
