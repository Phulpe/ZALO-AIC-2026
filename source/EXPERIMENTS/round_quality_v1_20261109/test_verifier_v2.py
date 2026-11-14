import unittest,json
from baseline_bridge import canonical_total,canonical_correction,BaselineBridge
class VerifierV2Tests(unittest.TestCase):
 def q(self,text):return {'prompt':json.dumps({'question':text,'as_of_date':'2026-01-05'}),'as_of_date':'2026-01-05','source_ref':'synthetic'}
 def test_supported_grammar(self):
  for metric in ['doanh thu','so luong ban','so don hang']:
   for period in ['hom qua','hom nay','thang truoc','tuan nay','nam truoc','thang 12','thang 1 nam 2026','ngay 03/01/2026']:
    with self.subTest(metric=metric,period=period):self.assertIsNotNone(canonical_total(self.q('tong '+metric+' trong '+period+' la bao nhieu?')))
 def test_reject_unconsumed_semantics(self):
  for text in ['doanh thu hom qua cua khach A','doanh thu hom qua khong gom hang huy','doanh thu hom qua so voi hom nay','doanh thu hom qua tren 100','doanh thu hom qua theo san pham','doanh thu hom qua trung binh','doanh thu ngay 32/01/2026','doanh thu thang 13 nam 2026','loi nhuan hom qua','doanh thu hom qua va hom nay']:
   with self.subTest(text=text):self.assertIsNone(canonical_total(self.q(text)))
 def test_gold_does_not_select_target(self):
  q=self.q('doanh thu hom qua');self.assertEqual(canonical_correction(q,{},123),canonical_correction(q,{},999))
 def test_relative_date_anchored_to_question(self):
  p=canonical_total(self.q('doanh thu hom qua'));self.assertEqual(p['time'],{'kind':'yesterday'})
 def test_wrong_metric_and_period_rejected(self):
  from test_core import fixture
  q=self.q('doanh thu thang 1 nam 2026');b=BaselineBridge(None,None,{'synthetic':fixture()});p=canonical_total(q)
  self.assertTrue(b.semantic(q,p))
  self.assertFalse(b.semantic(q,{**p,'metric':'orders'}))
  self.assertFalse(b.semantic(q,{**p,'time':{'kind':'month','value':'2025-12'}}))
