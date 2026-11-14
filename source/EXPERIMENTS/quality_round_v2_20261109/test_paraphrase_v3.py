import unittest,json
from baseline_bridge import canonical_total
class ParaphraseTests(unittest.TestCase):
 def q(self,s):return {'prompt':json.dumps({'question':s}),'as_of_date':'2026-02-28'}
 def test_equivalent(self):
  for s in ['cho toi biet tong doanh thu thang 1 nam 2026 la bao nhieu','trong thang 1 nam 2026 tong doanh thu la bao nhieu','tinh doanh thu thang 1 nam 2026']:
   self.assertEqual(canonical_total(self.q(s))['metric'],'revenue')
 def test_orders(self):self.assertEqual(canonical_total(self.q('co bao nhieu don hang hom qua'))['metric'],'orders')
 def test_filter_never_disappears(self):
  for s in ['cho toi biet doanh thu hom qua cua khach A','trong thang 1 nam 2026 doanh thu cua san pham B la bao nhieu','co bao nhieu don hang hom qua da huy']:
   self.assertIsNone(canonical_total(self.q(s)))
