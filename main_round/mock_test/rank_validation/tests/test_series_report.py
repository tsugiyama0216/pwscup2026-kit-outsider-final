import unittest

from rank_validation.series_report import MODELS, rank_rows


class SeriesRankingTests(unittest.TestCase):
    def rows(self):
        return [{"model": model, "index": (len(MODELS)-i)/10} for i,model in enumerate(MODELS)]

    def test_all_seven_candidates_are_required(self):
        with self.assertRaises(ValueError):
            rank_rows(self.rows()[:-1])
        values=self.rows(); values[-1]["model"]="Rv2"
        with self.assertRaises(ValueError):
            rank_rows(values)

    def test_tied_scores_share_rank(self):
        values=self.rows(); values[1]["index"]=values[0]["index"]
        result=rank_rows(values)
        self.assertEqual([r["rank"] for r in result[:3]], [1,1,3])

    def test_order_does_not_depend_on_input_sequence(self):
        self.assertEqual(rank_rows(self.rows()),rank_rows(list(reversed(self.rows()))))

    def test_nonfinite_score_is_not_ranked(self):
        for v in (float('nan'),float('inf'),-float('inf')):
            values=self.rows(); values[2]["index"]=v
            with self.assertRaises(ValueError):
                rank_rows(values)


if __name__ == '__main__':
    unittest.main()
