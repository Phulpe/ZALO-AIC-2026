import unittest
from audit_output import validate,audit
class OutputContractTests(unittest.TestCase):
 def test_number_is_not_boolean(self):
  with self.assertRaises(ValueError):validate({'id':'a','type':'number','answer':True})
 def test_nonfinite_rejected(self):
  for value in [float('nan'),float('inf')]:
   with self.assertRaises(ValueError):validate({'id':'a','type':'number','answer':value})
 def test_iso_real_calendar(self):
  validate({'id':'a','type':'date','answer':'2024-02-29'})
  validate({'id':'a','type':'date','answer':'2026-12'})
  for value in ['2026-02-29','2026-13','01/02/2026']:
   with self.assertRaises(ValueError):validate({'id':'a','type':'date','answer':value})
 def test_table_width(self):
  with self.assertRaises(ValueError):validate({'id':'a','type':'table','answer':{'columns':['n','v'],'rows':[['A']]}})
 def test_diagnostics_cannot_leak_into_submission(self):
  with self.assertRaises(ValueError):validate({'id':'a','type':'abstain','reason':'internal'})
 def test_missing_and_duplicate(self):
  result=audit([{'id':'a'},{'id':'b'}],[{'id':'a','type':'abstain'},{'id':'a','type':'abstain'}]);self.assertFalse(result['passed']);self.assertFalse(audit([{'id':'a'}],[{'id':['bad'],'type':'abstain'}])['passed'])
 def test_order(self):
  self.assertFalse(audit([{'id':'a'},{'id':'b'}],[{'id':'b','type':'abstain'},{'id':'a','type':'abstain'}])['passed'])
 def test_valid_full_submission(self):
  self.assertTrue(audit([{'id':'a'},{'id':'b'}],[{'id':'a','type':'number','answer':0},{'id':'b','type':'abstain'}])['passed'])
if __name__=='__main__':unittest.main()
