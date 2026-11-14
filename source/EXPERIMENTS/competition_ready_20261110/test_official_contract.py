"""Synthetic fixtures against the publicly published Evaluation specification.

These fixtures contain no competition questions, answers or SFT examples.
The official table was independently read on 2026-11-10; this is not a claim
that the organizers' private implementation is available or byte-identical.
"""
import sys, unittest, unicodedata
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent/'src/baseline/app'))
from scoring import matches, summarize


class PublishedContractTests(unittest.TestCase):
    def check(self, kind, predicted, reference, expected=True):
        self.assertEqual(matches({'type':kind,'answer':predicted},
                                 {'type':kind,'answer':reference}), expected)

    def test_number_exact_not_tolerant(self):
        self.check('number', 100.000001, 100, False)
        self.check('number', 100.0, 100)

    def test_percent_rounding_and_sign(self):
        self.check('percent', 12.341, 12.34)
        self.check('percent', -12.34, 12.34, False)

    def test_NFC_lowercase_and_punctuation(self):
        self.check('string', unicodedata.normalize('NFD', '  SẢN-PHẨM   MẪU '), 'sản phẩm mẫu')

    def test_diacritics_are_not_removed(self):
        self.check('string', 'san pham mau', 'sản phẩm mẫu', False)

    def test_reference_alias(self):
        self.assertTrue(matches({'type':'string','answer':'CODE17'},
                                {'type':'string','answer':'Tên giả lập','alias':['code17']}))

    def test_date_exact_and_month(self):
        self.check('date', '2026-02', '2026-02')
        self.check('date', '2026-02-01', '2026-02', False)

    def test_boolean_is_not_numeric(self):
        self.check('boolean', 1, True, False)
        self.check('boolean', False, False)

    def test_type_mismatch(self):
        self.assertFalse(matches({'type':'string','answer':'10'}, {'type':'number','answer':10}))

    def test_list_order_matters(self):
        self.check('list', ['mẫu b','mẫu a'], ['mẫu a','mẫu b'], False)
        self.check('list', ['MẪU A','mẫu b'], ['mẫu a','mẫu b'])

    def test_table_rows_and_named_columns_reordered(self):
        gold={'columns':['Tên','Số lượng'],'rows':[['mẫu a',2],['mẫu b',3]]}
        predicted={'columns':['SỐ LƯỢNG','tên'],'rows':[[3,'MẪU B'],[2,'mẫu a']]}
        self.check('table', predicted, gold)

    def test_table_positional_column_fallback(self):
        self.check('table', {'columns':['X','Y'],'rows':[['mẫu',7]]},
                   {'columns':['Tên','Lượng'],'rows':[['mẫu',7]]})

    def test_unanswerable_only_matches_same_type(self):
        self.assertTrue(matches({'type':'unanswerable'},{'type':'unanswerable'}))
        self.assertFalse(matches({'type':'number','answer':0},{'type':'unanswerable'}))

    def test_abstain_not_correct(self):
        self.assertFalse(matches({'type':'abstain'},{'type':'unanswerable'}))
        self.assertFalse(matches({'type':'abstain'},{'type':'abstain'}))

    def test_PA_plus_one_zero_minus_one(self):
        rows=[{'state':x} for x in ['correct','correct','wrong','abstain']]
        self.assertEqual(summarize(rows)['PA'], .25)


if __name__ == '__main__':
    unittest.main()
