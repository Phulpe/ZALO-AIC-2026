import unittest,json
from baseline_bridge import canonical_total,canonical_correction
class BridgeTests(unittest.TestCase):
    def question(self,text):return {'prompt':json.dumps({'question':text}),'as_of_date':'2026-11-08'}
    def test_monthly_total(self):
        p=canonical_total(self.question('Tổng doanh thu tháng 10 năm 2026 là bao nhiêu?'))
        self.assertIsNotNone(p);self.assertEqual(p['metric'],'revenue');self.assertEqual(p['time'],{'kind':'month','value':'2026-10'})
    def test_filter_not_silently_ignored(self):self.assertIsNone(canonical_total(self.question('Tổng doanh thu tháng 10 năm 2026 của khách A là bao nhiêu?')))
    def test_unsupported_correction(self):
        with self.assertRaises(ValueError):canonical_correction(self.question('Câu hỏi phức hợp'),{}, {'answer':999})
    def test_gold_independent_correction(self):
        q=self.question('Tổng doanh thu tháng 10 năm 2026 là bao nhiêu?')
        self.assertEqual(canonical_correction(q,{},1),canonical_correction(q,{},999))
    def test_real_executor_and_scorer(self):
        from test_core import fixture
        from baseline_bridge import BaselineBridge,matches
        q=self.question('<REDACTED_COMPETITION_QUESTION>');q.update(as_of_date='2026-01-31',source_ref='synthetic');ctx=json.loads(q['prompt']);ctx['as_of_date']=q['as_of_date'];q['prompt']=json.dumps(ctx)
        # Bridge execution uses no torch and no model load.
        b=BaselineBridge(None,None,{'synthetic':fixture()});p=canonical_total(q)
        self.assertTrue(b.semantic(q,p));self.assertTrue(matches(b.execute(q,p),{'type':'number','answer':300}))
    def test_gpu_smoke_targets_validated(self):
        from smoke_targets import targets
        self.assertEqual(len(targets()),2)
