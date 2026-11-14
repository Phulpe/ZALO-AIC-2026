"""Hoi quy: 'quy 03/2025', 'Q03/2025', 'quy 03' phai la QUY(3), khong bi doc thanh THANG(3)."""
import datetime as dt
import unittest
from datamate.hints import question_periods

D = dt.date(2025, 12, 20)


class QuarterZeroPad(unittest.TestCase):
    def codes(self, s):
        return [p.code for p in question_periods(s, D)]

    def test_with_year(self):
        for s in ("doanh thu quý 03/2025", "doanh thu Q03/2025", "doanh thu quý 03 năm 2025"):
            self.assertEqual(self.codes(s), ["QUY(3, 2025)"], s)

    def test_without_year(self):
        self.assertTrue(all(c.startswith("QUY(3") for c in self.codes("doanh thu quý 03")))

    def test_month_unchanged(self):
        self.assertEqual(self.codes("doanh thu tháng 03/2025"), ["THANG(3, 2025)"])
        self.assertEqual(self.codes("doanh thu quý 3/2025"), ["QUY(3, 2025)"])


if __name__ == "__main__":
    unittest.main()
