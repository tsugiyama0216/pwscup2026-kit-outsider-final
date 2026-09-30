"""Read-only aggregate audit of official distributions for v1 planning.

No predictions, record-level outputs, tokens, or network requests are produced.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from attack_workspace import verify_hashes


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def key(row, columns):
    return tuple(row[c] if c == "prefecture" else Decimal(row[c]) for c in columns)


def main():
    verified = verify_hashes()
    originals = ROOT.parent / "anonymization/team10/participant_data"
    schema = json.loads((originals / "schema.json").read_text(encoding="utf-8-sig"))
    qi = schema["aia_qi_columns"]
    columns = schema["submitted_c_columns"]
    b = read_csv(originals / "B.csv")
    bs = read_csv(originals / "B_self.csv")
    bg = read_csv(originals / "A_bg.csv")
    bs_by_id = {r["record_id"]: r for r in bs}
    assert all(key(r, columns) == key(bs_by_id[r["record_id"]], columns) for r in b)
    lookup = defaultdict(list)
    for row in b:
        lookup[key(row, qi)].append(row)
    self_challenge = read_csv(ROOT / "targets/aia_challenge_10.csv")
    matched = [r for r in self_challenge if key(r, qi) in lookup]
    match_sizes = Counter(len(lookup[key(r, qi)]) for r in matched)
    matched_b = [lookup[key(r, qi)][0] for r in matched if len(lookup[key(r, qi)]) == 1]
    matched_self = [bs_by_id[r["record_id"]] for r in matched_b]
    matched_keys = {key(r, qi) for r in matched}
    self_balance = {}
    for grouping in ("prefecture_sex", "prefecture_sex_5year_hypothesis"):
        members, controls = Counter(), Counter()
        for row in self_challenge:
            stratum = (row["prefecture"], row["sex"])
            if grouping.endswith("hypothesis"):
                stratum += (int(Decimal(row["age"])) // 5,)
            (members if key(row, qi) in matched_keys else controls)[stratum] += 1
        all_strata = set(members) | set(controls)
        self_balance[grouping] = {
            "strata": len(all_strata),
            "unequal_member_control_strata": sum(members[s] != controls[s] for s in all_strata),
        }
    self_report = {
        "rows": len(self_challenge),
        "exact_qi_matched_rows": len(matched),
        "exact_qi_unmatched_rows": len(self_challenge) - len(matched),
        "matched_candidate_multiplicities": dict(match_sizes),
        "distinct_b_candidates": len({r["record_id"] for r in matched_b}),
        "matched_onset_count": sum(int(r["onset"]) for r in matched_self),
        "matched_true_rare_count": sum(int(r["is_rare"]) for r in matched_self),
        "matched_true_rare_onset_count": sum(int(r["is_rare"]) * int(r["onset"]) for r in matched_self),
        "matched_death_count": sum(int(r["death"]) for r in matched_self),
        "matching_balance_checks": self_balance,
        "comparison": "Exact decimal equality on all seven public attributes; no tolerance or count fitting.",
    }
    targets = json.loads((ROOT / "targets/mia_columns.json").read_text(encoding="utf-8-sig"))["target_columns"]
    b_full = {key(r, columns) for r in b}
    bg_full = {key(r, columns) for r in bg}
    reports = []
    cross_challenge = defaultdict(set)
    for target in sorted(map(int, targets)):
        c = read_csv(ROOT / f"targets/C_{target}.csv")
        challenge = read_csv(ROOT / f"targets/aia_challenge_{target}.csv")
        for row in challenge:
            cross_challenge[key(row, qi)].add(target)
        previous = [{key(row, [col])[0] for row in c[:-1]} for col in columns]
        novel_cols = [col for col, values in zip(columns, previous) if key(c[-1], [col])[0] not in values]
        strata = Counter((r["prefecture"], r["sex"], int(Decimal(r["age"])) // 5) for r in challenge)
        strata_without_age = Counter((r["prefecture"], r["sex"]) for r in challenge)
        reports.append({
            "target": target,
            "last_row_novel_columns": novel_cols,
            "last_row_full_duplicates_in_previous": sum(key(r, columns) == key(c[-1], columns) for r in c[:-1]),
            "c_exact_full_matches_to_own_b": sum(key(r, columns) in b_full for r in c),
            "c_exact_full_matches_to_background": sum(key(r, columns) in bg_full for r in c),
            "challenge_exact_qi_matches_to_own_b": sum(key(r, qi) in lookup for r in challenge),
            "hypothetical_5year_strata": len(strata),
            "hypothetical_5year_strata_odd_counts": sum(n % 2 for n in strata.values()),
            "hypothetical_5year_strata_with_2_rows": sum(n == 2 for n in strata.values()),
            "prefecture_sex_strata_odd_counts": sum(n % 2 for n in strata_without_age.values()),
        })
    report = {
        "purpose": "Aggregate evidence for v1 planning; not an attack candidate or official evaluation.",
        "input_verification": verified,
        "self_challenge": self_report,
        "own_b_true_rare_count": sum(int(r["is_rare"]) for r in bs),
        "own_b_fpg_at_least_126_count": sum(Decimal(r["FPG"]) >= 126 for r in b),
        "cross_challenge_qi_keys_in_multiple_targets": sum(len(s) > 1 for s in cross_challenge.values()),
        "targets": reports,
        "limitations": [
            "Exact matching is a data observation, not access to organizer truth.",
            "Five-year strata are a tested hypothesis; parity does not prove the official matching algorithm.",
            "A last row made only of pre-existing column values is consistent with the hot-deck notice, not proof of provenance.",
            "No token or row IDs are read into outputs. No prediction files or submission packages are created.",
        ],
    }
    out = ROOT / "local-output/v1-planning-20260930/public_clues_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
