"""Independent fixtures for rulebook semantics; no public examples or IDs."""
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import pandas as pd
from test_core import fixture,plan
from planner import answer,parse_plan,context,_number
from legacy_engine import date_series,number_series,load_frames,prepare_shop,mapping_for,time_bounds,table_kind
from v2_contracts import canonical_time,ContractError,select_rules


class TimeContracts(unittest.TestCase):
 def test_missing_day_never_uses_host_clock(self):
  with self.assertRaises(ContractError):canonical_time({'kind':'day'})
 def test_month_day_mix_is_rejected(self):
  self.assertEqual(answer(fixture(),'Doanh thu tháng 1?','2026-01-31',plan(time={'kind':'month','value':'2026-01-01'}))['type'],'abstain')
 def test_impossible_day(self):
  with self.assertRaises(ContractError):canonical_time({'kind':'day','date':'2026-02-29'})
 def test_valid_leap_day(self):self.assertEqual(canonical_time({'kind':'date','value':'2024-02-29'}),{'kind':'day','date':'2024-02-29'})
 def test_reversed_range(self):
  with self.assertRaises(ContractError):canonical_time({'kind':'range','start':'2026-02-01','end':'2026-01-01'})
 def test_rolling_boolean_not_integer(self):
  with self.assertRaises(ContractError):canonical_time({'kind':'rolling_days','n':True})
 def test_rolling_must_be_positive(self):
  with self.assertRaises(ContractError):canonical_time({'kind':'rolling_days','n':0})
 def test_explicit_quarter(self):self.assertEqual(canonical_time({'kind':'quarter','value':'2024-Q1'})['end'],'2024-03-31')
 def test_year_alias(self):self.assertEqual(canonical_time({'kind':'year','value':'2024'})['end'],'2024-12-31')
 def test_relative_extra_fields_rejected(self):
  with self.assertRaises(ContractError):canonical_time({'kind':'today','date':'2026-01-01'})
 def test_legacy_month_arity_guard(self):
  with self.assertRaises(ValueError):
   # The planner wrapper is the public contract; legacy raises a domain error.
   canonical_time({'kind':'month','value':'2026-01-01'})
 def test_future_documented_noncoverage(self):
  self.assertEqual(answer(fixture(),'Doanh thu tháng 2/2026?','2026-01-31',plan(time={'kind':'month','value':'2026-02'})),{'type':'unanswerable'})
 def test_partial_period_uses_available_data(self):
  self.assertEqual(answer(fixture(),'Doanh thu giai đoạn?','2026-01-31',plan(time={'kind':'range','start':'2026-01-01','end':'2026-02-28'}))['answer'],300)
 def test_no_documented_end_does_not_infer_unanswerable(self):
  s=fixture();s.readme='No explicit publication boundary.'
  self.assertNotEqual(answer(s,'Doanh thu tháng 2?','2026-01-31',plan(time={'kind':'month','value':'2026-02'}))['type'],'unanswerable')


class SemanticRegression(unittest.TestCase):
 def test_amount_threshold_cannot_be_ignored(self):
  self.assertEqual(answer(fixture(),'Số đơn từ 500 nghìn trở lên?','2026-01-31',plan(metric='orders'))['type'],'abstain')
 def test_symbol_comparator_cannot_be_ignored(self):
  self.assertEqual(answer(fixture(),'Doanh thu các đơn > 500000?','2026-01-31',plan())['type'],'abstain')
 def test_multiple_measures_not_answered_by_one_total(self):
  self.assertEqual(answer(fixture(),'Doanh thu và số đơn?','2026-01-31',plan())['type'],'abstain')
 def test_date_range_is_not_a_money_threshold(self):
  p=plan(metric='orders',time={'kind':'range','start':'2026-01-01','end':'2026-01-31'})
  self.assertEqual(answer(fixture(),'Số đơn từ ngày 01/01/2026 đến ngày 31/01/2026?','2026-01-31',p)['answer'],2)
 def test_quantity_question_does_not_accept_revenue(self):
  self.assertEqual(answer(fixture(),'Tổng số lượng bán?','2026-01-31',plan())['type'],'abstain')
 def test_distinct_customers_not_counted_as_orders(self):
  self.assertEqual(answer(fixture(),'Có bao nhiêu khách đã mua?','2026-01-31',plan(metric='orders'))['type'],'abstain')
 def test_share_not_replaced_by_cancellation_rate(self):
  self.assertEqual(answer(fixture(),'Tỷ trọng doanh thu tiền mặt?','2026-01-31',plan(metric='cancellation_rate',answer_type='percent'))['type'],'abstain')
 def test_gross_profit_not_replaced_by_sales(self):
  self.assertEqual(answer(fixture(),'Lãi gộp bao nhiêu?','2026-01-31',plan())['type'],'abstain')
 def test_explicit_month_disagrees_with_plan(self):
  self.assertEqual(answer(fixture(),'Doanh thu tháng 2/2026?','2026-01-31',plan())['type'],'abstain')
 def test_relative_day_disagrees_with_plan(self):
  self.assertEqual(answer(fixture(),'Doanh thu hôm qua?','2026-01-31',plan())['type'],'abstain')
 def test_explicit_day_disagrees_with_plan(self):
  self.assertEqual(answer(fixture(),'Doanh thu ngày 02/01/2026?','2026-01-31',plan())['type'],'abstain')
 def test_ranking_compares_different_units(self):
  s=fixture();s.lines.loc[1,'base_unit']='kg'
  self.assertEqual(answer(s,'Mặt hàng nào bán chạy nhất?','2026-01-31',plan(metric='quantity',group_by='product',limit=1,answer_type='string'))['answer'],'Hàng A')
 def test_mixed_units_not_summed_as_one_quantity(self):
  s=fixture();s.lines.loc[1,'base_unit']='kg'
  self.assertEqual(answer(s,'Tổng số lượng?','2026-01-31',plan(metric='quantity'))['type'],'abstain')
 def test_customer_id_without_name_does_not_crash(self):
  s=fixture();s.lines=s.lines.drop(columns='customer');s.lines['customer_id']=['K1','K2']
  self.assertEqual(answer(s,'Khách nào mua nhiều tiền nhất?','2026-01-31',plan(group_by='customer',answer_type='string',limit=1))['answer'],'K2')
 def test_customer_identity_uses_ids_even_same_names(self):
  s=fixture();s.lines['customer_id']=['K1','K2'];s.lines['customer']=['Cùng Tên','Cùng Tên']
  r=answer(s,'Doanh thu theo khách?','2026-01-31',plan(group_by='customer',answer_type='table'))
  self.assertEqual(sorted(row[1] for row in r['answer']['rows']),[100,200])
 def test_valid_id_not_discarded_by_generic_name(self):
  s=fixture();s.lines['customer_id']=['K1','K2']
  r=answer(s,'Doanh thu theo khách?','2026-01-31',plan(group_by='customer',answer_type='table'))
  self.assertEqual(len(r['answer']['rows']),2)
 def test_customer_name_writing_variants_are_one_customer(self):
  s=fixture();s.lines['customer']=['Lê An','LÊ-AN']
  r=answer(s,'Doanh thu theo khách?','2026-01-31',plan(group_by='customer',answer_type='table'))
  self.assertEqual(r['answer']['rows'],[['Lê An',300]])
 def test_fractional_quantity_not_rounded_to_cents(self):
  s=fixture();s.lines['base_qty']=[0.001,0.002]
  self.assertEqual(answer(s,'Tổng số cái?','2026-01-31',plan(metric='quantity',unit='cái'))['answer'],0.003)
 def test_decimal_sum_is_not_binary_float_artifact(self):
  s=fixture();s.lines['base_qty']=[0.1,0.2]
  self.assertEqual(answer(s,'Tổng số cái?','2026-01-31',plan(metric='quantity',unit='cái'))['answer'],0.3)
 def test_identifier_case_is_not_silently_merged(self):
  s=fixture();s.lines['product_id']=['a','A'];s.products=pd.DataFrame()
  p=plan(filters=[{'field':'product_id','op':'eq','value':'a'}])
  self.assertEqual(answer(s,'Doanh thu mặt hàng có mã a?','2026-01-31',p)['answer'],100)
 def test_cancelled_and_returned_excluded_shipping_kept(self):
  s=fixture();s.lines=pd.concat([s.lines]*2,ignore_index=True)
  s.lines['_status']=['completed','cancelled','returned','shipping']
  s.lines['order_id']=['O1','O2','O3','O4']
  self.assertEqual(answer(s,'Tổng doanh thu?','2026-01-31',plan())['answer'],300)
 def test_cancellation_denominator_includes_every_status(self):
  s=fixture();s.lines=pd.concat([s.lines]*2,ignore_index=True)
  s.lines['_status']=['completed','cancelled','returned','shipping'];s.lines['order_id']=['O1','O2','O3','O4']
  self.assertEqual(answer(s,'Tỷ lệ đơn huỷ?','2026-01-31',plan(metric='cancellation_rate',answer_type='percent'))['answer'],25)
 def test_order_total_does_not_become_line_revenue(self):
  self.assertEqual(mapping_for(['Tổng tiền'],'orders',{}),{'Tổng tiền':'total'})
 def test_explicit_line_revenue_mapping_is_preserved(self):
  self.assertEqual(mapping_for(['Tiền hàng'],'orders',{'tienhang':'line_total'}),{'Tiền hàng':'line_total'})


class IngestionAndContext(unittest.TestCase):
 def test_invoice_filename_not_a_product_catalog(self):self.assertEqual(table_kind('BanHang_HoaDon.xlsx'),'orders')
 def test_invoice_detail_keeps_line_role(self):self.assertEqual(table_kind('BanHang_ChiTietHoaDon.xlsx'),'sales')
 def test_product_export_keeps_catalog_role(self):self.assertEqual(table_kind('BanHang_HangHoa.xlsx'),'products')
 def test_unknown_sheet_falls_back_to_filename(self):self.assertEqual(table_kind('BanHang_HoaDon.xlsx','Export'),'orders')
 def test_specific_sheet_still_overrides_workbook(self):self.assertEqual(table_kind('BanHang_HoaDon.xlsx','HangHoa'),'products')
 def test_invoice_short_id_maps_for_join(self):self.assertEqual(mapping_for(['MaHD'],'sales',{}),{'MaHD':'order_id'})
 def test_transaction_id_maps_for_join(self):self.assertEqual(mapping_for(['Mã giao dịch'],'sales',{}),{'Mã giao dịch':'order_id'})
 def test_date_is_joined_from_header_on_invoice_id(self):
  with TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'README.md').write_text('Dữ liệu có đến hết 31/01/2026.')
   pd.DataFrame({'Mã hóa đơn':['H1'],'Ngày bán':['01/01/2026'],'Tổng tiền hàng':[100],'Trạng thái':['Hoàn thành']}).to_excel(p/'BanHang_HoaDon.xlsx',index=False)
   pd.DataFrame({'MaHD':['H1'],'Tên hàng':['Hàng A'],'Mã hàng':['A'],'Số lượng':[1],'Thành tiền':[100]}).to_excel(p/'BanHang_ChiTietHoaDon.xlsx',index=False)
   shop=prepare_shop(p)
   self.assertEqual(shop.lines.date.iloc[0],pd.Timestamp('2026-01-01'))
   self.assertEqual(shop.orders.revenue.iloc[0],100)
 def test_currency_suffixes_and_unicode_minus(self):
  s=number_series(pd.Series(['1,5 triệu','120k','−25.000 đ','(2.000 VNĐ)']))
  self.assertEqual(s.tolist(),[1500000,120000,-25000,-2000])
 def test_boolean_not_a_money_amount(self):self.assertTrue(pd.isna(number_series(pd.Series([True])).iloc[0]))
 def test_malformed_arithmetic_is_not_a_number(self):self.assertTrue(pd.isna(number_series(pd.Series(['2 x 100k'])).iloc[0]))
 def test_weekday_prefix_and_sheet_year(self):self.assertEqual(date_series(pd.Series(['Thứ hai, ngày 12/03']),'Năm 2024').iloc[0],pd.Timestamp('2024-03-12'))
 def test_day_without_year_requires_evidence(self):self.assertTrue(pd.isna(date_series(pd.Series(['12/03'])).iloc[0]))
 def test_day_in_sheet_inherits_sheet_year(self):self.assertEqual(date_series(pd.Series(['12/03']),'Bán hàng 2024').iloc[0],pd.Timestamp('2024-03-12'))
 def test_day_only_in_vietnamese_month_sheet(self):self.assertEqual(date_series(pd.Series([12]),'Thg 3 2024').iloc[0],pd.Timestamp('2024-03-12'))
 def test_late_rule_section_is_retrieved(self):
  text='# Quy tắc\n'+'\n'.join('- Quy tắc khác.' for _ in range(20))+'\n| Giá vốn | Giá vốn ghi trên dòng bán. |'
  rules=select_rules(text,'Giá vốn và lãi gộp?',__import__('legacy_engine').norm)
  self.assertTrue(any('dòng bán' in row for row in rules))
 def test_rule_is_not_cut_mid_sentence(self):
  long='| Doanh thu | '+('bỏ đơn huỷ và trả hàng; '*10)+'không cộng VAT. |'
  rules=select_rules(long,'doanh thu?',__import__('legacy_engine').norm)
  self.assertTrue(rules[0].endswith('không cộng VAT.'))
 def test_context_contains_relevant_table_rules(self):
  ctx=context(fixture(),'Doanh thu?','2026-01-31','# Quy tắc\n| Doanh thu | Bỏ đơn huỷ và trả hàng. |')
  self.assertTrue(ctx['shared_rules'])
 def test_order_id_not_forward_filled_across_sources(self):
  with TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'README.md').write_text('Dữ liệu có đến hết 31/01/2026.')
   pd.DataFrame({'Ngày bán':['01/01/2026'],'Mã đơn hàng':['O1'],'Tên hàng':['Hàng A'],'Số lượng':[1],'Thành tiền':[100]}).to_csv(p/'BanHangA.csv',index=False)
   pd.DataFrame({'Ngày bán':['02/01/2026'],'Mã đơn hàng':[None],'Tên hàng':['Hàng B'],'Số lượng':[1],'Thành tiền':[200]}).to_csv(p/'BanHangB.csv',index=False)
   shop=prepare_shop(p)
   self.assertTrue(pd.isna(shop.lines.loc[shop.lines['product'].eq('Hàng B'),'order_id'].iloc[0]))


if __name__=='__main__':unittest.main()
