import unittest
from audit_eval import score_saved


class AuditLeakGuardTests(unittest.TestCase):
    def test_missing_predictions_do_not_open_gold(self):
        with self.assertRaises(ValueError):
            score_saved(['fixture_a'], [], lambda:self.fail('Gold opened'), None)

    def test_duplicate_predictions_do_not_open_gold(self):
        with self.assertRaises(ValueError):
            score_saved(['fixture_a','fixture_b'], [{'id':'fixture_a'},{'id':'fixture_a'}], lambda:self.fail('Gold opened'), None)

    def test_duplicate_frozen_ids_do_not_open_gold(self):
        with self.assertRaises(ValueError):
            score_saved(['fixture_a','fixture_a'], [], lambda:self.fail('Gold opened'), None)

    def test_wrong_gold_split_is_rejected(self):
        with self.assertRaises(ValueError):
            score_saved(['fixture_a'], [{'id':'fixture_a'}], lambda:{'fixture_b':{}}, None)

    def test_scoring_only_after_complete_predictions(self):
        order=[]
        def read():
            order.append('gold');return {'fixture_a':{},'fixture_b':{}}
        def match(row, label):
            self.assertEqual(order[0], 'gold');return row['type']=='number'
        result=score_saved(['fixture_a','fixture_b'], [{'id':'fixture_a','type':'number'},
            {'id':'fixture_b','type':'abstain'}], read, match)
        self.assertEqual((result['correct'],result['wrong'],result['abstain']),(1,0,1))
        self.assertEqual(result['metric'],.5)


if __name__ == '__main__':
    unittest.main()
