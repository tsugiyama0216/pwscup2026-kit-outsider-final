"""Aggregate-only exact subset audit; excludes tokens and all old candidates."""
from __future__ import annotations

import csv
import itertools
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from attack_workspace import verify_hashes


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [
            {k: (v if k == "prefecture" else Decimal(v)) for k, v in r.items()}
            for r in csv.DictReader(stream)
        ]


def key(r, cols):
    return tuple(r[c] for c in cols)


def main():
    verified = verify_hashes()
    original = ROOT.parent / "anonymization/team10/participant_data"
    schema = json.loads((original / "schema.json").read_text(encoding="utf-8-sig"))
    b, bs, bg = [read(original / f"{stem}.csv") for stem in ["B", "B_self", "A_bg"]]
    qi = schema["aia_qi_columns"]
    self_keys = {key(r, qi) for r in read(ROOT / "targets/aia_challenge_10.csv")}
    private = [key(r, qi) in self_keys for r in b]
    assert sum(private) == 163
    true_rare_by_id = {r["record_id"]: int(r["is_rare"]) for r in bs}
    rare = [true_rare_by_id[r["record_id"]] for r in b]
    continuous = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
    lab = continuous[1:]
    subsets = list(itertools.combinations(continuous, 2)) + list(itertools.combinations(continuous, 3))
    subsets += [tuple(lab), tuple(continuous), tuple(qi), tuple(schema["submitted_c_columns"])]
    bgsets = {cols: {key(r, cols) for r in bg} for cols in subsets}
    bytarget = []
    targets = json.loads((ROOT / "targets/mia_columns.json").read_text(encoding="utf-8-sig"))["target_columns"]
    for target in sorted(map(int, targets)):
        if target == 10:
            continue
        c = read(ROOT / f"targets/C_{target}.csv")
        subs = []
        for cols in subsets:
            ck = Counter(key(r, cols) for r in c)
            hit = [key(r, cols) in ck for r in b]
            subs.append({
                "columns": list(cols),
                "b_known_private_hits": sum(h and p for h, p in zip(hit, private)),
                "b_remaining_hits": sum(h and not p for h, p in zip(hit, private)),
                "b_remaining_true_rare_hits": sum(h and not p and rr for h, p, rr in zip(hit, private, rare)),
                "c_rows_matching_background": sum(key(r, cols) in bgsets[cols] for r in c),
                "c_unique_subset_keys": len(ck),
            })
        event_records = [r for r in c if r["onset"] or r["death"]]
        event_reports = []
        for cols in [
            ("time", "onset", "death"),
            ("time", "onset", "death", "sex"),
            ("time", "onset", "death", "sex", "age"),
            ("time", "onset", "death", "sex", "prefecture"),
            ("time", "onset", "death", "sex", "age", "prefecture"),
        ]:
            for mode in ["decimal_exact", "rounded_day"]:
                def event_key(row):
                    return tuple((row[col] * Decimal("365.25")).to_integral_value() if col == "time" and mode == "rounded_day" else row[col] for col in cols)
                ck = Counter(event_key(r) for r in event_records)
                bgkeys = {event_key(r) for r in bg if r["onset"] or r["death"]}
                hit = [bool(r["onset"] or r["death"]) and event_key(r) in ck for r in b]
                event_reports.append({
                    "columns": list(cols), "mode": mode,
                    "b_known_private_hits": sum(h and p for h, p in zip(hit, private)),
                    "b_remaining_hits": sum(h and not p for h, p in zip(hit, private)),
                    "b_remaining_true_rare_hits": sum(h and not p and rr for h, p, rr in zip(hit, private, rare)),
                    "c_event_rows_matching_background": sum(event_key(r) in bgkeys for r in event_records),
                })
        bytarget.append({"target": target, "subsets": subs, "event_fingerprints": event_reports})
    report = {
        "purpose": "Aggregate supplementary diagnosis only; exact matches are observations, not membership labels.",
        "verified": verified,
        "group_sizes": {
            "b_known_private": sum(private), "b_remaining": len(b) - sum(private),
            "known_private_event": sum(p and bool(r["onset"] or r["death"]) for p, r in zip(private, b)),
            "remaining_event": sum(not p and bool(r["onset"] or r["death"]) for p, r in zip(private, b)),
            "remaining_true_rare": sum(rr and not p for rr, p in zip(rare, private)),
            "background_rows": len(bg),
        },
        "targets": bytarget,
        "limitations": [
            "Remaining B rows are unlabeled, not known members.",
            "Known private rows are a quota-selected subset, not a representative negative sample.",
            "Decimal equality is exact; rounded-day equality is a separate coarse diagnostic.",
            "Subset coincidences and background matches alone do not establish provenance.",
            "Only aggregate counts are saved; no record IDs, private values, predictions, or credentials are emitted.",
        ],
    }
    out = Path(__file__).with_name("subset_clues.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verified": verified, "group_sizes": report["group_sizes"], "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
