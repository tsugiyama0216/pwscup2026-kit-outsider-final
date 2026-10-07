"""Fixed-real-release mock cases with private truth and newly synthesized controls.

This module never imports an attack. A_bg rare labels are detector proxies, and
the controls' rare labels define this synthetic experiment, not official truth.
"""
from __future__ import annotations

from collections import Counter
import csv
from decimal import Decimal
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .common import (ROOT, PARTICIPANT, FEATURES, QI, cfg, ensure_new_output,
                     rare_mask, read_csv, sha256, write_json)

NUMERIC = ["BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
QI_NUMERIC = ["SBP", "TG", "HDL", "ALT"]
MEMBER_QUOTAS = {(0, 0): 138, (0, 1): 12, (1, 0): 9, (1, 1): 4}
MIN_QI_DISTANCE = 0.12


def _keys(frame: pd.DataFrame, columns) -> list[tuple]:
    """Numeric-equivalent exact keys, including CSV decimal precision."""
    return [tuple(str(v) if column == "prefecture" else Decimal(str(v))
                  for column, v in zip(columns, row))
            for row in frame[list(columns)].itertuples(index=False, name=None)]


def _exact_csv(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return pd.DataFrame(list(csv.DictReader(stream)))


def _official_members(b: pd.DataFrame, b_strings: pd.DataFrame,
                      challenge_strings: pd.DataFrame) -> np.ndarray:
    lookup = {}
    for index, key in enumerate(_keys(b_strings, QI)):
        if key in lookup:
            raise ValueError("B has ambiguous duplicate AIA QI keys")
        lookup[key] = index
    members = [lookup[key] for key in _keys(challenge_strings, QI) if key in lookup]
    if len(members) != 163 or len(set(members)) != 163:
        raise ValueError("Expected exactly 163 unique official AIA members in B")
    selected = b.iloc[members]
    quotas = Counter(zip(selected.is_rare.astype(int), selected.onset.astype(int)))
    if dict(quotas) != MEMBER_QUOTAS:
        raise ValueError(f"Official member rare/onset quotas differ: {dict(quotas)}")
    return np.asarray(members, dtype=int)


def _sample_mia(b, background, member_indices, schema, rng, rare_positive):
    available = np.ones(len(b), dtype=bool)
    available[member_indices] = False
    positives = []
    for rare, count in ((0, 26 - rare_positive), (1, rare_positive)):
        pool = np.flatnonzero(available & (b.is_rare.to_numpy(int) == rare))
        if len(pool) < count:
            raise ValueError(f"Not enough MIA positive records in rare={rare}")
        positives.extend(rng.choice(pool, count, replace=False).tolist())
    # Identity here is the exact complete feature vector, not the independently
    # assigned record_id values in B and A_bg.
    b_keys = set(_keys(b, FEATURES))
    allowed = np.asarray([key not in b_keys for key in _keys(background, FEATURES)])
    background_rare = np.asarray(rare_mask(background, schema), dtype=bool)
    negatives = []
    for rare, count in ((0, 1023 - (24 - rare_positive)), (1, 24 - rare_positive)):
        pool = np.flatnonzero(allowed & (background_rare == bool(rare)))
        if len(pool) < count:
            raise ValueError(f"Not enough nonoverlapping background records in rare_proxy={rare}")
        negatives.extend(rng.choice(pool, count, replace=False).tolist())
    rows = pd.concat([b.iloc[positives][FEATURES], background.iloc[negatives][FEATURES]],
                     ignore_index=True)
    truth = pd.DataFrame({"member": [1] * 26 + [0] * 1023,
                          "is_rare": np.r_[b.iloc[positives].is_rare.to_numpy(int),
                                            background_rare[negatives].astype(int)]})
    source = pd.DataFrame({"source": ["B"] * 26 + ["A_bg"] * 1023,
                           "source_row": positives + negatives})
    permutation = rng.permutation(1049)
    rows, truth, source = [frame.iloc[permutation].reset_index(drop=True)
                           for frame in (rows, truth, source)]
    for frame in (rows, truth, source):
        frame.insert(0, "record_id", np.arange(1049))
    reduced = background.drop(background.index[negatives]).copy()
    return rows, truth, source, reduced, np.asarray(negatives)


def _donor_pool(donors, member):
    """Keep rare proxy and outcome fixed; explicitly record sparse-cell backoff.

    Requested controls always retain all five matching stratum fields. Only the
    donor pool broadens; no requested output count or matching quota is relaxed.
    """
    rare = donors.is_rare.to_numpy(int) == int(member.is_rare)
    onset = donors.onset.to_numpy(int) == int(member.onset)
    sex = donors.sex.to_numpy(int) == int(member.sex)
    prefecture = donors.prefecture.to_numpy() == member.prefecture
    age = ((donors.age.to_numpy(int) - 40) // 5) == ((int(member.age) - 40) // 5)
    levels = [("exact", rare & onset & sex & prefecture & age),
              ("without_prefecture", rare & onset & sex & age),
              ("without_prefecture_age", rare & onset & sex),
              ("rare_onset", rare & onset)]
    for label, mask in levels:
        indices = np.flatnonzero(mask)
        if len(indices) >= 2:
            return indices, label
    raise ValueError("At least two background donors required in every rare/onset stratum")


def _synthetic_controls(members, donors, all_b, all_background, schema, configuration, rng):
    donors = donors.copy()
    donors["is_rare"] = np.asarray(rare_mask(donors, schema), dtype=int)
    numeric = donors[NUMERIC].to_numpy(float)
    covariance = np.cov(numeric, rowvar=False)
    covariance = .95 * covariance + .05 * np.diag(np.maximum(np.diag(covariance), 1e-6))
    qi_scale = all_background[QI_NUMERIC].std(ddof=0).to_numpy(float)
    reference = pd.concat([all_b[FEATURES], all_background[FEATURES]], ignore_index=True)
    reference_numeric = reference[QI_NUMERIC].to_numpy(float) / qi_scale
    forbidden_qi = set(_keys(reference, QI))
    forbidden_full = set(_keys(reference, FEATURES))
    horizon = float(configuration["onset"]["horizon_years"])
    rows, provenance, minimum_distances = [], [], []
    backoffs = Counter()
    for control_index, (_, member) in enumerate(members.iterrows()):
        pool, backoff = _donor_pool(donors, member)
        accepted = False
        rejected = Counter()
        for attempt in range(10000):
            # Independent empirical draws may select the same donor. Forcing
            # different tail donors can average opposite extremes into a normal
            # point and make the requested rare stratum impossible to sample.
            first, second = rng.choice(pool, 2, replace=True)
            left, right = donors.iloc[first], donors.iloc[second]
            row = left[FEATURES].copy()
            weight = rng.uniform(.80, .98)
            measurements = (weight * left[NUMERIC].to_numpy(float)
                            + (1 - weight) * right[NUMERIC].to_numpy(float)
                            + .15 * rng.multivariate_normal(np.zeros(6), covariance))
            for column, value in zip(NUMERIC, measurements):
                row[column] = float(np.clip(value, *schema["ranges"][column]))
            low = 40 + ((int(member.age) - 40) // 5) * 5
            row["age"] = int(rng.integers(low, min(low + 4, schema["age_range"][1]) + 1))
            row["sex"], row["prefecture"] = int(member.sex), str(member.prefecture)
            row["onset"] = int(member.onset)
            row["smoking"] = int(left.smoking)
            # Outcomes come from background donors, never from the matched B
            # member. Censoring has no jitter; events are inside the horizon.
            row["death"] = 0 if row["onset"] else int(left.death)
            time_donor = None
            if row["onset"] == 0 and row["death"] == 0:
                row["time"] = horizon
            else:
                event_pool = pool[donors.iloc[pool].death.to_numpy(int) == row["death"]]
                time_donor = int(rng.choice(event_pool))
                row["time"] = float(np.clip(donors.iloc[time_donor].time + rng.normal(0, .30),
                                               1 / 365.25, horizon - 1 / 365.25))
            candidate = pd.DataFrame([row], columns=FEATURES)
            if bool(np.asarray(rare_mask(candidate, schema))[0]) != bool(member.is_rare):
                rejected["rare_proxy"] += 1
                continue
            qi_key, full_key = _keys(candidate, QI)[0], _keys(candidate, FEATURES)[0]
            if qi_key in forbidden_qi or full_key in forbidden_full:
                rejected["collision"] += 1
                continue
            # Reject near-copies of *any* original health QI vector. This is a
            # conservative distance check, irrespective of age/sex/prefecture.
            vector = candidate[QI_NUMERIC].to_numpy(float)[0] / qi_scale
            distance = float(np.sqrt(((reference_numeric - vector) ** 2).sum(axis=1).min()))
            if distance < MIN_QI_DISTANCE:
                rejected["near_copy"] += 1
                continue
            accepted = True
            break
        if not accepted:
            raise ValueError(f"Control {control_index}: unable to satisfy matching/uniqueness constraints; "
                             f"backoff={backoff}, pool={len(pool)}, rare={int(member.is_rare)}, rejections={dict(rejected)}")
        forbidden_qi.add(qi_key)
        forbidden_full.add(full_key)
        reference_numeric = np.vstack([reference_numeric, vector])
        row["is_rare"] = int(member.is_rare)
        rows.append(row)
        provenance.append({"control_id": control_index, "matched_B_record_id": int(member.record_id),
                           "donor_A_bg_row_1": int(donors.index[first]),
                           "donor_A_bg_row_2": int(donors.index[second]),
                           "outcome_A_bg_row": None if time_donor is None else int(donors.index[time_donor]),
                           "donor_pool_size": len(pool), "donor_backoff": backoff,
                           "generation_attempts": attempt + 1,
                           "minimum_scaled_qi_distance": distance,
                           "rare_label_source": "synthetic stratum; A_bg detector proxy"})
        minimum_distances.append(distance)
        backoffs[backoff] += 1
    controls = pd.DataFrame(rows).reset_index(drop=True)
    controls.insert(0, "control_id", np.arange(len(controls)))
    for name in ("age", "sex", "smoking", "onset", "death", "is_rare"):
        controls[name] = controls[name].astype(int)
    quotas = Counter(zip(controls.is_rare, controls.onset))
    if dict(quotas) != MEMBER_QUOTAS:
        raise ValueError("Synthetic control quota validation failed")
    fields = ["prefecture", "sex", "age_band", "onset", "is_rare"]
    left, right = members.copy(), controls.copy()
    for frame in (left, right):
        frame["age_band"] = (frame.age.astype(int) - 40) // 5
    if Counter(map(tuple, left[fields].to_numpy())) != Counter(map(tuple, right[fields].to_numpy())):
        raise ValueError("Synthetic controls do not preserve the requested matching strata")
    return controls, pd.DataFrame(provenance), {
        "donor_backoff_counts": dict(backoffs),
        "minimum_scaled_qi_distance": min(minimum_distances),
        "near_copy_rejection_threshold": MIN_QI_DISTANCE,
        "rare_label_basis": "Synthetic assigned stratum checked with the public detector; not official hidden labels",
    }


def _csv(frame, path):
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _select_releases(selection_seed, r_count):
    if not 0 <= r_count <= 12:
        raise ValueError("r_count must be between 0 and 12")
    base = ROOT / "main_round" / "anonymization" / "team10"
    rng = np.random.default_rng(selection_seed)
    selected = sorted(rng.choice(np.arange(1, 13), r_count, replace=False).tolist())
    pairs = [("M1_v5", base / "M" / "M1_v5" / "C.csv"),
             ("S1", base / "S" / "S1" / "C.csv")]
    pairs.extend((f"R{number}", base / "R" / f"R{number}" / "C.csv") for number in selected)
    return [{"name": name, "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
            for name, path in pairs]


def generate(output: Path, seed=2026100701, count=3, selection_seed=20261007, r_count=10) -> dict:
    """Write a reproducible suite. Public files contain no private provenance."""
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    output = ensure_new_output(Path(output))
    releases = _select_releases(selection_seed, r_count)
    b_path, bg_path = PARTICIPANT / "B_self.csv", PARTICIPANT / "A_bg.csv"
    source_b_path = PARTICIPANT / "B.csv"
    challenge_path = ROOT / "main_round" / "attack" / "targets" / "aia_challenge_10.csv"
    schema_path = PARTICIPANT / "schema.json"
    b, bg = read_csv(b_path), read_csv(bg_path)
    if len(b) != 1049 or len(bg) < 1025:
        raise ValueError("Unexpected participant source sizes")
    b_plain = read_csv(source_b_path)
    if not b[(["record_id"] + list(FEATURES))].equals(b_plain[["record_id"] + list(FEATURES)]):
        raise ValueError("B_self and B records do not match exactly")
    schema = json.loads(schema_path.read_text(encoding="utf-8-sig"))
    configuration = cfg()
    member_indices = _official_members(b, _exact_csv(source_b_path), _exact_csv(challenge_path))
    members = b.iloc[member_indices].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    cases = []
    for index in range(count):
        case_id = f"case_{index + 1:03d}"
        case = output / case_id
        public, private = case / "public", case / "private"
        public.mkdir(parents=True)
        private.mkdir()
        candidates, mia_truth, mia_sources, reduced, negative_indices = _sample_mia(
            b, bg, member_indices, schema, rng, index % 3)
        controls, control_sources, diagnostics = _synthetic_controls(
            members, reduced, b, bg, schema, configuration, rng)
        queries = pd.concat([members[QI], controls[QI]], ignore_index=True)
        truth = pd.concat([members[["time", "onset", "is_rare"]],
                           controls[["time", "onset", "is_rare"]]], ignore_index=True)
        truth.insert(0, "label", ["member"] * 163 + ["control"] * 163)
        sources = pd.DataFrame({"source": ["B"] * 163 + ["synthetic"] * 163,
                                "source_id": members.record_id.tolist() + controls.control_id.tolist()})
        permutation = rng.permutation(326)
        queries, truth, sources = [frame.iloc[permutation].reset_index(drop=True)
                                   for frame in (queries, truth, sources)]
        for frame in (queries, truth, sources):
            frame.insert(0, "challenge_row_id", np.arange(326))
        _csv(candidates, public / "B.csv")
        _csv(reduced[FEATURES].reset_index(drop=True), public / "A_bg.csv")
        _csv(queries, public / "aia_challenge_0.csv")
        shutil.copyfile(schema_path, public / "schema.json")
        write_json(public / "cfg.json", configuration)
        for filename, frame in (("mia_truth.csv", mia_truth), ("aia_truth.csv", truth),
                                ("controls.csv", controls), ("mia_provenance.csv", mia_sources),
                                ("aia_provenance.csv", sources),
                                ("control_provenance.csv", control_sources)):
            _csv(frame, private / filename)
        protocol = {
            "version": 1, "case_id": case_id, "case_sequence": index,
            "mia": {"candidates": 1049, "members": 26, "nonmembers": 1023,
                    "rare_total": 24, "rare_members": index % 3,
                    "positive_rare_basis": "B_self official birth label",
                    "negative_rare_basis": "A_bg public Mahalanobis detector proxy",
                    "excluded_A_bg_source_rows": sorted(map(int, negative_indices)),
                    "AIA_members_excluded_from_MIA_positives": True},
            "aia": {"members": 163, "controls": 163, "onset_per_class": 16,
                    "rare_per_class": 13, "rare_onset_per_class": 4,
                    "matching": "prefecture, sex, five-year age band, onset, rare stratum",
                    "member_truth": "Exact Decimal seven-QI match to the official own-team challenge",
                    "control_truth": "New synthetic background-donor outcomes, not hidden official controls",
                    "outcome_model": "Background event time donor plus 0.30y Gaussian jitter; administrative censoring at horizon",
                    "source_diagnostics": diagnostics},
            "isolation": "Each release is attacked in its own fresh process/public directory; no state transfer",
            "background_history_warning": "Removing rows does not untrain existing models or erase past background use during C creation",
        }
        write_json(private / "protocol.json", protocol)
        cases.append({"id": case_id, "path": case_id})
    sources = [b_path, source_b_path, bg_path, schema_path, challenge_path]
    manifest = {
        "version": 1, "seed": seed, "count": count, "selection_seed": selection_seed,
        "releases": releases, "cases": cases,
        "sources": {path.relative_to(ROOT).as_posix(): sha256(path) for path in sources},
        "files": {path.relative_to(output).as_posix(): sha256(path)
                  for path in sorted(output.rglob("*")) if path.is_file()},
        "limitations": [
            "Official scoring arithmetic is reused; these cases and floors are not official hidden tests or leaderboard predictions.",
            "AIA controls are synthetic and can be distinguishable from real members; matching and distance checks do not prove distribution equivalence.",
            "A_bg rare labels and synthetic control rare strata are proxies, not official birth labels.",
            "MIA negatives must be excluded from training/cache use, not only the current public background. Prior access cannot be undone.",
            "Existing releases may have used all of A_bg during creation; background removal does not change their history.",
            "All releases derive from the same B; each run must have fresh attack state. Repeated cases reuse one finite population.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    return manifest
