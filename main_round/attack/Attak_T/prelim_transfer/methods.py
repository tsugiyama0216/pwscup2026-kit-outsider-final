"""Reuse historical mathematical helpers without running historical CLIs."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import attack_workspace as workspace
from local_eval.data import FEATURES, QI, rare_mask

LEGACY = workspace.ROOT.parents[1] / "prelim" / "workspace"
AIA_METHODS = ("v4", "independent", "structured", "density",
               "complement_structured", "complement_density")


def assert_official_origins():
    for name, module in list(sys.modules.items()):
        if name == "pwscup2026" or name.startswith("pwscup2026."):
            path = getattr(module, "__file__", None)
            if path and not Path(path).resolve().is_relative_to(workspace.PACKAGE.resolve()):
                raise RuntimeError(f"Historical official package leaked into main runtime: {name}")


def load_legacy():
    workspace.load_runtime()  # Load the pinned main kit BEFORE historical imports.
    old_path = sys.path.copy()
    try:
        sys.path.insert(0, str(LEGACY))
        names = {
            "v1": "attack_phase.generate_submission",
            "v3": "attack_phase_v3.generate_submission_v3",
            "v4": "attack_phase_v4.generate_submission_v4",
            "conformal": "attack_phase_v4.mia_private_conformal",
            "independent": "attack_phase_daybreak_independent.generate_submission_daybreak_independent",
            "structured": "attack_phase_v5.aia_structured_ensemble",
            "density": "attack_phase_v6.aia_global_density",
        }
        modules = {key: importlib.import_module(name) for key, name in names.items()}
    finally:
        sys.path[:] = old_path
    assert_official_origins()
    return modules


def source_hashes():
    paths = {Path(m.__file__).resolve() for name, m in list(sys.modules.items())
             if name.startswith("attack_phase") and getattr(m, "__file__", None)}
    paths.update(Path(__file__).parent.glob("*.py"))
    paths.update((workspace.ROOT / "local_eval").glob("*.py"))
    paths.add(workspace.ROOT / "attack_workspace.py")
    return {str(p.relative_to(workspace.ROOT.parents[1])).replace("\\", "/"): workspace.sha256(p)
            for p in sorted(paths)}


def build_mia(modules, candidates, background, targets, self_challenge, schema,
              *, minimum=25, maximum=27, total=None, include_domias=True):
    """Exactly the selected v4 branch; omit only discarded experimental branches."""
    if len(targets) < 2:
        raise ValueError("The selected v4 MIA requires multiple simultaneous targets")
    if total is None:
        # 736 shared rows across 29 peers; only 22 peers are present here.
        # This is an explicit uniform allocation prior, not a known target count.
        total = round(736 * len(targets) / 29)
    private, detection = modules["v3"].discover_known_private_records(candidates, self_challenge)
    detected_rare = rare_mask(candidates, schema)
    reference, _, transport = modules["v1"].build_mia(
        candidates, background, targets, minimum=minimum, maximum=maximum,
        total=total, transport_epsilon=.35, include_domias=include_domias)
    cf = modules["conformal"]
    reference = cf.demote_known_private(reference, private)
    conformal, calibration = cf.build_private_conformal_mia(candidates, background, targets, private)
    fused = cf.reciprocal_rank_fusion(reference, conformal["conformal_margin_0p50"],
                                     private, rank_constant=10.)
    result = cf.demote_known_private(cf.preserve_subset_order(fused, reference, detected_rare), private)
    if not (result.loc[private, list(targets)].max() < result.loc[~private, list(targets)].min()).all():
        raise AssertionError("Inferred self-private rows were not strictly demoted")
    diagnostics = {"method": "preliminary_v4_selected_branch", "private_detection": detection,
        "public_detected_rare_rows": int(detected_rare.sum()), "capacity_prior_total": total,
        "capacity_prior_basis": "round(736 * number_of_present_peers / 29); not ground truth",
        "transport": transport, "calibration": calibration,
        "known_private_top_counts": modules["v3"]._private_top_counts(result, private),
        "known_private_basis": "Exact unique self-challenge QI matches plus contest private-block design"}
    return result, {"transport_private_reference": reference}, diagnostics


def complement(*frames):
    ordered = [f.sort_values(["target_k", "challenge_row_id"]).reset_index(drop=True) for f in frames]
    keys = ["target_k", "challenge_row_id"]
    if any(not ordered[0][keys].equals(f[keys]) for f in ordered[1:]):
        raise ValueError("Complement experts have different challenge keys")
    unanimous = np.logical_and.reduce([~np.isclose(f.time_hat.to_numpy(), 11.) for f in ordered])
    result = ordered[0].copy()
    result["time_hat"] = np.where(unanimous, 11., 9.5)
    return result


def build_aia(modules, background, targets, challenges):
    """Historical families with fixed parameters; no old target-number routing."""
    if set(targets) != set(challenges):
        raise ValueError("Target/challenge sets differ")
    releases = {k: c[FEATURES].copy() for k, c in targets.items()}
    queries = {k: q[["challenge_row_id", *QI]].copy() for k, q in challenges.items()}
    background = background[FEATURES].copy()
    v4, _, v4_detail = modules["v4"].build_aia_v4(background, releases, queries,
        main_fraction=.5, variant_fractions=[.5], abstain_time=11., horizon_time=10.,
        link_trust_distance=.20, baseline_abstain_time=modules["v1"].choose_low_hit_time(
            background, horizon=10., tau=.5), anomalous_event_rate=.50,
        anomalous_link_trust_distance=.08, integrity_auc_threshold=.58)
    independent, _, independent_detail = modules["independent"].build_aia_independent(
        background, releases, queries, member_count=163, residual_weight=.25,
        background_neighbours=48, integrity_auc_threshold=.60, horizon=10., abstain_time=11.)
    ensemble, ensemble_detail = modules["structured"].build_aia_v5_variants(
        v4, independent, releases, queries, member_count=163, abstain_time=11., horizon_time=10.)
    structured = ensemble["consensus_v4_copula__v4_event_margin"]
    density, density_detail = modules["density"].build_global_density_horizon(
        releases, queries, background, member_count=163, abstain_time=11., horizon_time=10.)
    variants = {"v4": v4, "independent": independent, "structured": structured, "density": density,
        "complement_structured": complement(v4, independent, structured),
        "complement_density": complement(v4, independent, density)}
    for name, frame in variants.items():
        values = frame.time_hat.to_numpy(float)
        if len(frame) != sum(map(len, queries.values())) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid AIA predictions: {name}")
    assert_official_origins()
    return variants, {"v4": v4_detail, "independent": independent_detail,
        "structured": ensemble_detail, "density": density_detail,
        "complement": "v7 complement of three active sets, edge time 9.5; two fixed routed experts"}
