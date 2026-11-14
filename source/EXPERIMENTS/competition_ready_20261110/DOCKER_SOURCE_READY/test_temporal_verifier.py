import unittest,json
from temporal_verifier import canonical_total,same_time

class TemporalVerifierTests(unittest.TestCase):
 def q(self,text):return {'prompt':json.dumps({'question':text,'as_of_date':'2024-03-01'}),'as_of_date':'2024-03-01'}
 def test_full_simple_grammar(self):
  for text in ['Giúp tôi tính tổng số đơn hàng hôm qua là bao nhiêu?','Vui lòng cho biết doanh thu ngày hôm qua','Trong hôm qua tổng số lượng hàng đã bán là bao nhiêu?','Hôm qua doanh thu bao nhiêu?','Có bao nhiêu đơn hàng hôm qua']:
   with self.subTest(text=text):self.assertEqual(canonical_total(self.q(text))['time'],{'kind':'yesterday'})
 def test_leftover_conditions_not_erased(self):
  for text in ['Doanh thu hôm qua của khách A','Doanh thu hôm qua không gồm hàng huỷ','Doanh thu hôm qua so với hôm nay','Doanh thu hôm qua trên 100','Doanh thu hôm qua theo sản phẩm','Số đơn hàng hôm qua trên 2 triệu','Doanh thu hôm qua trước 8 giờ','Doanh thu tháng 13 năm 2024','Doanh thu hôm qua và hôm nay']:
   with self.subTest(text=text):self.assertIsNone(canonical_total(self.q(text)))
 def test_equivalent_day_leap_anchor(self):
  self.assertTrue(same_time({'kind':'yesterday'},{'kind':'day','date':'2024-02-29'},'2024-03-01'))
  self.assertFalse(same_time({'kind':'today'},{'kind':'yesterday'},'2024-03-01'))
 def test_month_to_date_is_not_full_month(self):
  self.assertFalse(same_time({'kind':'this_month'},{'kind':'month','value':'2024-03'},'2024-03-15'))
  self.assertTrue(same_time({'kind':'last_month'},{'kind':'month','value':'2024-02'},'2024-03-15'))
 def test_invalid_time_fails(self):
  for t in [{'kind':'none'},{'kind':'day','date':'2024-02-30'},{'kind':'today','extra':1}]:
   self.assertFalse(same_time(t,{'kind':'yesterday'},'2024-03-01'))

if __name__=='__main__':unittest.main()
