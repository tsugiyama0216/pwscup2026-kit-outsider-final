"""Reconcile user-reported final feedback without inferring hidden row truth."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
TEAM = REPO / "submissions/prelim/team24"
TARGETS = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12",
           "13", "17", "18", "19", "20", "22", "23", "26", "29"}
ZERO_TARGETS = {"6", "7", "8", "9", "17", "18", "19"}
POSITIVE_SOURCES = {
    "v4": {"1", "2", "4", "5", "12", "29"},
    "independent": {"3", "13", "22", "23", "26"},
    "routed": {"10", "20"},
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def aia_blocks(path: Path) -> dict[str, list[tuple[int, float]]]:
    # The production recipe compares pandas-canonical numeric blocks. Historical
    # decimal strings can lose a last digit when the generator reads/writes CSV;
    # this is not a claim of byte identity for AIA (unlike the copied MIA file).
    blocks: dict[str, list[tuple[int, float]]] = {}
    for row in pd.read_csv(path).itertuples(index=False):
        blocks.setdefault(str(int(row.target_k)), []).append(
            (int(row.challenge_row_id), float(row.time_hat)))
    return {target: sorted(rows) for target, rows in blocks.items()}


class OfficialFinalFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.final = read_json(ROOT / "official_feedback/guarded_final.json")
        previous = REPO / "attack_phase_v6/official_feedback"
        cls.previous = {
            "v4": read_json(previous / "v4.json"),
            "independent": read_json(previous / "probe_independent.json"),
            "routed": read_json(previous / "routed_guided_probe.json"),
        }

    def test_complete_user_report_and_constants(self) -> None:
        final = self.final
        self.assertEqual(final["candidate"], "v9_guarded")
        self.assertEqual(final["source"], "user_reported")
        self.assertEqual(set(final["mia_per_target"]), TARGETS)
        self.assertEqual(set(final["aia_per_target"]), TARGETS)
        self.assertEqual(final["constants"], {
            "low_fpr": 0.01, "tau_time": 0.5, "aia_rare_se_z": 1.64,
            "mia_band_edges": [0.25, 0.5, 0.75]})
        for value in [*final["mia"].values(), *final["aia"].values(),
                      *final["aia_per_target"].values()]:
            self.assertTrue(math.isfinite(value) and 0 <= value <= 1)
        self.assertEqual(final["aia"], {
            "r_time_all": 0.1658175911401578, "r_time_rare": 0.1,
            "p_m": 0.5328220858895707, "p_c": 0.4138036809815951,
            "p_m_rare": 0.4076923076923077, "p_c_rare": 0.3192307692307693})

    def test_displayed_metrics_match_full_precision(self) -> None:
        self.assertEqual(self.final["score_1_displayed"], 0.3407)
        for index, section, field in (
            (2, "mia", "tpr_at_low_fpr"), (3, "mia", "tpr_rare"),
            (4, "aia", "r_time_all"), (5, "aia", "r_time_rare"),
        ):
            self.assertEqual(self.final[f"score_{index}_displayed"],
                             round(self.final[section][field], 4))

    def test_mia_unchanged_metrics_and_artifact(self) -> None:
        for section in ("mia", "mia_per_target"):
            self.assertEqual(self.final[section], self.previous["v4"][section])
        current = (TEAM / "attack-candidate-v9-guarded/F_mia.csv").read_bytes()
        previous = (TEAM / "attack-candidate-v4/F_mia.csv").read_bytes()
        self.assertEqual(current, previous)
        self.assertEqual(hashlib.sha256(current).hexdigest().upper(),
                         "D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B")

    def test_positive_target_scores_replay_measured_winners(self) -> None:
        positives = set().union(*POSITIVE_SOURCES.values())
        self.assertEqual(len(positives), 13)
        self.assertEqual(positives, TARGETS - ZERO_TARGETS)
        for source, targets in POSITIVE_SOURCES.items():
            for target in targets:
                score = self.final["aia_per_target"][target]
                self.assertEqual(score, self.previous[source]["aia_per_target"][target])
                self.assertEqual(score, max(record["aia_per_target"][target]
                                            for record in self.previous.values()))

    def test_whole_target_canonical_artifacts_match_recipe(self) -> None:
        path = TEAM / "attack-candidate-v9-guarded/F_aia.csv"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest().upper(),
                         "63944D5FAA2C08E5E2E40415FFBB16B3153B27EBDD052CD68D7ADB59A403A6E0")
        final = aia_blocks(path)
        self.assertEqual(set(final), TARGETS)
        for rows in final.values():
            self.assertEqual([row[0] for row in rows], list(range(326)))
        recipe = {
            "attack-candidate-v4": POSITIVE_SOURCES["v4"],
            "attack-candidate-daybreak-independent": POSITIVE_SOURCES["independent"],
            "attack-candidate-v6-routed-guided-probe": POSITIVE_SOURCES["routed"],
            "attack-candidate-v7-feedback-stack": ZERO_TARGETS,
        }
        for directory, targets in recipe.items():
            source = aia_blocks(TEAM / directory / "F_aia.csv")
            for target in targets:
                self.assertEqual(final[target], source[target])

    def test_five_previously_zero_targets_improved(self) -> None:
        historical_zeros = {
            target for target in TARGETS
            if all(record["aia_per_target"][target] == 0
                   for record in self.previous.values())}
        self.assertEqual(historical_zeros, ZERO_TARGETS)
        self.assertEqual({target for target in ZERO_TARGETS
                          if self.final["aia_per_target"][target] > 0},
                         {"6", "8", "17", "18", "19"})

    def test_target_mean_and_improvement_decomposition(self) -> None:
        scores = self.final["aia_per_target"]
        positives = math.fsum(scores[t] for t in TARGETS - ZERO_TARGETS) / 20
        formerly_zero = math.fsum(scores[t] for t in ZERO_TARGETS) / 20
        total = math.fsum(scores.values()) / 20
        previous = math.fsum(self.previous["routed"]["aia_per_target"].values()) / 20
        self.assertAlmostEqual(positives, 0.1506204125415067, places=15)
        self.assertAlmostEqual(formerly_zero, 0.032124655148639644, places=15)
        self.assertAlmostEqual(total, 0.18274506769014637, places=15)
        self.assertAlmostEqual(total, positives + formerly_zero, places=15)
        self.assertAlmostEqual(positives - previous, 0.008139260499509277, places=15)
        self.assertAlmostEqual(total - previous, 0.04026391564814894, places=15)
        self.assertNotAlmostEqual(total, self.final["score_4_displayed"] +
                                  self.final["score_5_displayed"], places=4)

    def test_same_mia_reconciles_rounded_official_totals(self) -> None:
        previous = self.previous["independent"]
        old_mia = previous["score_1_displayed"] - math.fsum(
            previous["aia_per_target"].values()) / 20
        new_aia = math.fsum(self.final["aia_per_target"].values()) / 20
        new_mia = self.final["score_1_displayed"] - new_aia
        half_width = 0.00005
        self.assertLess(abs(new_mia - old_mia), 2 * half_width)
        self.assertEqual(round(old_mia + new_aia, 4), self.final["score_1_displayed"])
        self.assertAlmostEqual(new_mia, 0.15795493230985364, places=15)
        self.assertAlmostEqual(new_mia - old_mia, 0.00004344438682834, places=15)
        self.assertAlmostEqual(self.final["score_1_displayed"] -
                               self.previous["routed"]["score_1_displayed"], 0.0403)


if __name__ == "__main__":
    unittest.main()
