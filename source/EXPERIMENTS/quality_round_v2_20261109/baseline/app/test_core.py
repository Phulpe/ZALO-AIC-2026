import unittest,json
from types import SimpleNamespace
import pandas as pd
from scoring import matches,text_key
from planner import answer,validate_plan,parse_plan,context

def fixture():
 f=pd.DataFrame({'date':pd.to_datetime(['2026-01-01','2026-01-02']),'order_id':['O1','O2'],'product_id':['A','B'],'product':['Hàng A','Hàng B'],'base_qty':[10,2],'base_unit':['cái','cái'],'line_total':[100,200],'_status':['completed','completed'],'qty':[10,2],'unit':['cái','cái'],'customer':['Khách lẻ','Khách lẻ']})
 return SimpleNamespace(lines=f,orders=f.copy(),products=f[['product_id','product']].copy(),tables={},start=pd.Timestamp('2026-01-01'),end=pd.Timestamp('2026-01-31'),readme='Dữ liệu có đến hết 31/01/2026.',audit=[],has_orders=True)
def plan(**kw):
 p={'metric':'revenue','time':{'kind':'month','value':'2026-01'},'filters':[],'group_by':None,'order':'desc','limit':None,'unit':None,'answer_type':'number','unresolved':[]};p.update(kw);return p
class CoreTests(unittest.TestCase):
 def test_spaces(self):
  self.assertNotEqual(text_key('AB'),text_key('A B'));self.assertEqual(text_key(' A--B  '),'a b');self.assertNotEqual(text_key('đá'),text_key('da'))
 def test_percent(self):self.assertTrue(matches({'type':'percent','answer':12.345},{'type':'percent','answer':12.35}))
 def test_alias(self):self.assertTrue(matches({'type':'string','answer':'SP01'},{'type':'string','answer':'Hàng A','alias':['SP01']}))
 def test_list_order(self):self.assertFalse(matches({'type':'list','answer':['B','A']},{'type':'list','answer':['A','B']}))
 def test_table(self):self.assertTrue(matches({'type':'table','answer':{'columns':['v','n'],'rows':[[2,'B'],[1,'A']]}},{'type':'table','answer':{'columns':['n','v'],'rows':[['A',1],['B',2]]}}))
 def test_revenue_ranking(self):self.assertEqual(answer(fixture(),'Sản phẩm nào có doanh thu cao nhất tháng 1/2026?','2026-01-31',plan(group_by='product',limit=1,answer_type='string'))['answer'],'Hàng B')
 def test_wrong_metric(self):self.assertEqual(answer(fixture(),'Sản phẩm nào có doanh thu cao nhất?','2026-01-31',plan(metric='quantity',group_by='product',limit=1,answer_type='string'))['type'],'abstain')
 def test_unconsumed_product(self):self.assertEqual(answer(fixture(),'Doanh thu Hàng A tháng 1/2026?','2026-01-31',plan())['type'],'abstain')
 def test_filter(self):self.assertEqual(answer(fixture(),'Doanh thu Hàng A tháng 1/2026?','2026-01-31',plan(filters=[{'field':'product_id','op':'eq','value':'A'}]))['answer'],100)
 def test_unknown_filter(self):self.assertEqual(answer(fixture(),'Doanh thu tháng này?','2026-01-31',plan(filters=[{'field':'product_id','op':'eq','value':'Z'}]))['type'],'abstain')
 def test_partial_coverage(self):self.assertEqual(answer(fixture(),'Doanh thu giai đoạn?','2026-01-31',plan(time={'kind':'range','start':'2026-01-01','end':'2026-02-01'}))['answer'],300)
 def test_true_zero(self):self.assertEqual(answer(fixture(),'Doanh thu ngày 03/01/2026?','2026-01-31',plan(time={'kind':'day','date':'2026-01-03'}))['answer'],0)
 def test_bad_date(self):
  s=fixture();s.lines.loc[0,'date']=pd.NaT;self.assertEqual(answer(s,'Doanh thu tháng này?','2026-01-31',plan())['type'],'abstain')
 def test_json(self):self.assertIsNone(parse_plan('{"metric":"revenue"}'))
 def test_schema(self):
  with self.assertRaises(ValueError):validate_plan(plan(filters=[{'field':'invented','op':'eq','value':'A'}]))
 def test_context(self):self.assertEqual(context(fixture(),'Doanh thu Hàng A?','2026-01-31','')['entity_candidates'][0]['value'],'A')
 def test_unresolved(self):self.assertEqual(answer(fixture(),'Tính số liệu?','2026-01-31',plan(unresolved=['unknown']))['type'],'abstain')
 def test_units(self):self.assertEqual(answer(fixture(),'Bán bao nhiêu cái?','2026-01-31',plan(metric='quantity',unit='cái'))['answer'],12)
if __name__=='__main__':unittest.main()
