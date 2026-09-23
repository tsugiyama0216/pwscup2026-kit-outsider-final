"""One-time, ledgered epsilon=1/3/10 releases; never import private scoring here."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import dp_copula
import dp_cox

HERE = Path(__file__).resolve().parent
EPSILONS = (1, 3, 10)
TOTAL_EPSILON = Fraction(14)
POSTPROCESS_SEED = 20260918


class ReleaseBlocked(RuntimeError):
    """An ambiguous/reserved release may not access private data again."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, content: dict, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        # The file's mere existence marks the reservation, even after a crash
        # before its contents are fully written. Never overwrite reservations.
        with path.open("xb") as stream:
            stream.write(_canonical(content))
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("wb") as stream:
        stream.write(_canonical(content))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_config(config: dict) -> dict:
    """Validate all three release plans before any reservation/private read."""
    if config.get("version") != 1 or config.get("postprocess_seed") != POSTPROCESS_SEED:
        raise ValueError("Config version must be 1 and postprocess_seed must be 20260918")
    if set(config.get("releases", {})) != {str(e) for e in EPSILONS}:
        raise ValueError("Frozen config must specify exactly epsilon 1, 3 and 10")
    plans = {}
    for eps in EPSILONS:
        item = config["releases"][str(eps)]
        # A string avoids unnoticed binary multiplication before Fraction.
        if not isinstance(item.get("copula_fraction"), str):
            raise ValueError("copula_fraction must be an exact rational string")
        fraction = Fraction(item["copula_fraction"])
        if not 0 < fraction <= 1:
            raise ValueError("copula_fraction must be in (0,1]")
        if not isinstance(item.get("bins", 8), int) or not 2 <= item.get("bins", 8) <= 32:
            raise ValueError("bins must be an integer from 2 through 32")
        if set(item.get("sample", {})) - set(dp_copula.DEFAULT_SAMPLE):
            raise ValueError("Unknown sample option")
        sample = {**dp_copula.DEFAULT_SAMPLE, **item.get("sample", {})}
        for name in ["group_prior", "marginal_prior_noise_multiplier", "prefecture_prior"]:
            if not isinstance(sample[name], (int, float)) or not math.isfinite(sample[name]) or sample[name] < 0:
                raise ValueError(f"Invalid sample option {name}")
        for name in ["public_correlation_strength", "public_radial_strength"]:
            if not isinstance(sample[name], (int, float)) or not 0 <= sample[name] <= 1:
                raise ValueError(f"Invalid sample option {name}")
        for name in ["enforce_rare_partition", "enforce_public_support", "stratified"]:
            if not isinstance(sample[name], bool):
                raise ValueError(f"{name} must be boolean")
        calibration = item.get("cox_calibration", {})
        if calibration.get("strata", "rare") not in ("rare", "none"):
            raise ValueError("Cox calibration strata must be rare or none")
        for section, allowed in [
            ("cox_public", {"quantization", "ridge", "clip_quantile"}),
            ("cox_postprocess", {"damping", "max_standardized_step", "prior_variance_multiplier"}),
            ("cox_calibration", {"iterations", "learning_rate", "max_corr_mean_shift", "max_corr_tail_shift", "strata"}),
        ]:
            if set(item.get(section, {})) - allowed:
                raise ValueError(f"Unknown {section} option")
        pub = {"quantization": 128, "ridge": .01, "clip_quantile": .99, **item.get("cox_public", {})}
        if not isinstance(pub["quantization"], int) or pub["quantization"] < 16:
            raise ValueError("Cox quantization must be an integer >=16")
        if not 0 < pub["ridge"] < float("inf") or not 0 < pub["clip_quantile"] <= 1:
            raise ValueError("Invalid public Cox fit parameters")
        post = {"damping": 1, "max_standardized_step": 1, "prior_variance_multiplier": 1,
                **item.get("cox_postprocess", {})}
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in post.values()):
            raise ValueError("Cox postprocess parameters must be finite and nonnegative")
        cal = {"iterations": 16, "learning_rate": .7, "max_corr_mean_shift": .02,
               "max_corr_tail_shift": .04, **item.get("cox_calibration", {})}
        if not isinstance(cal["iterations"], int) or cal["iterations"] < 0 or not 0 < cal["learning_rate"] <= 1:
            raise ValueError("Invalid Cox calibration iterations/rate")
        if any(not math.isfinite(cal[n]) or cal[n] < 0 for n in ["max_corr_mean_shift", "max_corr_tail_shift"]):
            raise ValueError("Invalid Cox calibration correlation guard")
        copula = Fraction(eps) * fraction
        cox = Fraction(eps) - copula
        if copula + cox != eps:
            raise AssertionError("Exact component composition failed")
        # Verify module query allocations also compose exactly before release.
        detail = dp_copula.privacy_plan(copula)
        if sum(Fraction(q["epsilon_allocated_exact"]) for q in detail["queries"]) != copula:
            raise AssertionError("Copula accounting mismatch")
        plans[eps] = {"epsilon_exact": str(eps), "copula_epsilon_exact": str(copula),
                      "cox_epsilon_exact": str(cox), "copula_fraction": str(fraction)}
    if sum(Fraction(p["epsilon_exact"]) for p in plans.values()) != TOTAL_EPSILON:
        raise AssertionError("The three releases must compose to epsilon14")
    return plans


def _reservation_history(ledger_dir: Path) -> set[int]:
    events = ledger_dir / "events.jsonl"
    reserved = set()
    if events.exists():
        try:
            for line in events.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if event.get("event") == "reserved":
                    reserved.add(int(event["accounting"]["epsilon_exact"]))
        except (ValueError, KeyError, TypeError) as exc:
            raise ReleaseBlocked("Incomplete or corrupt append-only ledger requires manual review") from exc
    if reserved - set(EPSILONS):
        raise ReleaseBlocked("Unexpected prior budget reservation")
    return reserved


def _freeze(config_path: Path, data_dir: Path, output_dir: Path) -> dict:
    # Never call _sha on B, B_self, utility_ref, or any other private file.
    source = {name: _sha(HERE / name) for name in ["release.py", "dp_copula.py", "dp_cox.py"]}
    public = {name: _sha(data_dir / name) for name in ["A_bg.csv", "schema.json"]}
    versions = {}
    for name in ["numpy", "pandas", "scipy", "lifelines"]:
        versions[name] = importlib.metadata.version(name)
    identity = {"format": "S_Copula_frozen_release_v1", "config_sha256": _sha(config_path),
                "source_sha256": source, "public_input_sha256": public,
                "library_versions": versions, "data_directory": str(data_dir.resolve()),
                "output_directory": str(output_dir.resolve()),
                "public_n": dp_copula.PRIVATE_N, "epsilon_releases": list(EPSILONS),
                "total_epsilon_exact": str(TOTAL_EPSILON), "delta": 0,
                "postprocess_seed": POSTPROCESS_SEED}
    identity["freeze_sha256"] = hashlib.sha256(_canonical(identity)).hexdigest()
    return identity


@contextmanager
def _exclusive_run(ledger_dir: Path):
    ledger_dir.mkdir(parents=True, exist_ok=True)
    lock = ledger_dir / "active.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ReleaseBlocked("Another run or a stale crash lock exists; inspect the ledger, do not retry private fits") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "created_utc": _now()}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        lock.unlink(missing_ok=True)


def _append_event(ledger_dir: Path, event: dict) -> None:
    record = {"time_utc": _now(), **event}
    with (ledger_dir / "events.jsonl").open("ab") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_existing_model(path: Path, freeze: dict, plan: dict) -> dict:
    model = json.loads(path.read_text(encoding="utf-8"))
    if model.get("freeze_sha256") != freeze["freeze_sha256"] or model.get("accounting") != plan:
        raise ReleaseBlocked("Saved model does not match the immutable freeze/accounting")
    if model.get("format") != "S_Copula_combined_DP_model_v1":
        raise ReleaseBlocked("Saved model format is invalid")
    return model


def _generate(model: dict, item: dict) -> tuple[pd.DataFrame, dict]:
    synthetic = dp_copula.sample(model["copula"], n=dp_copula.PRIVATE_N,
                                 seed=POSTPROCESS_SEED, **item.get("sample", {}))
    trace = {"method": "no_cox_calibration", "selection_input": "frozen_public_configuration"}
    if model.get("cox") is not None:
        eps = Fraction(model["accounting"]["cox_epsilon_exact"])
        target = dp_cox.postprocess_cox_update(model["cox"]["noisy_score_integer"], model["public_cox"],
                   epsilon=eps, public_n=dp_copula.PRIVATE_N, **item.get("cox_postprocess", {}))
        settings = dict(item.get("cox_calibration", {}))
        stratum = settings.pop("strata", "rare")
        if stratum == "rare":
            detector = model["copula"]["schema"]["rare_detector"]
            strata = (dp_copula._radii(synthetic, detector) >= float(detector["band"]["lo"])).astype(int)
        else:
            strata = None
        synthetic, trace = dp_cox.calibrate_synthetic_cox(synthetic, target, model["public_cox"],
                                                         strata=strata, **settings)
    # All checks below observe only the DP model/generated output.
    if len(synthetic) != dp_copula.PRIVATE_N or synthetic.isna().any().any():
        raise ReleaseBlocked("DP postprocessing produced an invalid row count or missing values")
    if not np.isfinite(synthetic[dp_copula.NUMERIC]).all().all():
        raise ReleaseBlocked("DP postprocessing produced nonfinite numeric output")
    return synthetic, trace


def run(config_path: Path, data_dir: Path, output_dir: Path, *,
        selected: tuple[int, ...] = EPSILONS, ledger_dir: Path | None = None) -> list[dict]:
    """Reserve then release each budget once. A saved model can be resumed.

    ledger_dir is injected by unit tests; the CLI fixes it alongside this code.
    Deleting/copying ledgers or changing this code cannot reset privacy spending.
    """
    config_path, data_dir, output_dir = map(Path, (config_path, data_dir, output_dir))
    ledger_dir = Path(ledger_dir) if ledger_dir is not None else HERE / "release_ledger"
    if not selected or len(set(selected)) != len(selected) or any(e not in EPSILONS for e in selected):
        raise ValueError("selected epsilons must be distinct members of 1,3,10")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plans = validate_config(config)
    freeze = _freeze(config_path, data_dir, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    private_cache = None
    with _exclusive_run(ledger_dir):
        freeze_path = ledger_dir / "freeze.json"
        if freeze_path.exists():
            if json.loads(freeze_path.read_text(encoding="utf-8")) != freeze:
                raise ReleaseBlocked("Config, source, public inputs, versions or paths changed after freeze")
        else:
            # A reservation without a freeze is ambiguous and must not restart.
            if list(ledger_dir.glob("epsilon_*.reservation.json")):
                raise ReleaseBlocked("Existing reservations without a complete freeze require review")
            _atomic_json(freeze_path, freeze, exclusive=True)
        public = pd.read_csv(data_dir / "A_bg.csv")
        schema = json.loads((data_dir / "schema.json").read_text(encoding="utf-8"))
        historical_reservations = _reservation_history(ledger_dir)
        for eps in selected:
            plan, item = plans[eps], config["releases"][str(eps)]
            folder = output_dir / f"eps{eps}"
            model_path, csv_path = folder / "model.json", folder / "C.csv"
            manifest_path = folder / "manifest.json"
            reservation = ledger_dir / f"epsilon_{eps}.reservation.json"
            if eps in historical_reservations and not reservation.exists():
                raise ReleaseBlocked("A prior reservation is recorded but its file is missing; private retry blocked")
            if reservation.exists():
                if not model_path.exists():
                    raise ReleaseBlocked(f"epsilon {eps} is conservatively spent/reserved but has no complete saved model; private retry blocked")
                model = _load_existing_model(model_path, freeze, plan)
                if manifest_path.exists() and csv_path.exists():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest["model_sha256"] != _sha(model_path) or manifest["C_sha256"] != _sha(csv_path):
                        raise ReleaseBlocked("Existing DP artifacts differ from their saved hashes")
                    reports.append({"epsilon": eps, "status": "already_complete", "directory": str(folder)})
                    continue
                _append_event(ledger_dir, {"event": "resume_postprocessing_only", "epsilon_exact": str(eps),
                                           "model_sha256": _sha(model_path)})
            else:
                if model_path.exists() or csv_path.exists() or manifest_path.exists():
                    raise ReleaseBlocked("Output exists without its reservation; refusing overwrite or a new private fit")
                # Build all public nuisance parameters BEFORE spending budget.
                public_model = None
                if Fraction(plan["cox_epsilon_exact"]) > 0:
                    categories = sorted(str(v) for v in public.prefecture.dropna().unique())
                    defaults = {col: float(pd.to_numeric(public[col], errors="coerce").median()) for col in dp_copula.NUMERIC}
                    clean_public = dp_copula._prepare(public, schema, categories, defaults)
                    public_model = dp_cox.build_public_cox_model(clean_public, **item.get("cox_public", {}))
                _atomic_json(reservation, {"created_utc": _now(), "state": "reserved_conservatively_spent",
                                          "freeze_sha256": freeze["freeze_sha256"], "accounting": plan}, exclusive=True)
                _append_event(ledger_dir, {"event": "reserved", "accounting": plan})
                try:
                    if private_cache is None:
                        private_cache = pd.read_csv(data_dir / "B.csv")
                    copula = dp_copula.fit(private_cache, public, schema, Fraction(plan["copula_epsilon_exact"]),
                                           bins=item.get("bins", 8))
                    cox, mechanism = None, None
                    if public_model is not None:
                        # Identical public-domain preprocessing before private Cox
                        # prevents undefined/overflowing scores outside bounds.
                        clean_private = dp_copula._prepare(private_cache, schema, copula["categories"], copula["public_defaults"])
                        def noise(values, sensitivity, epsilon):
                            nonlocal mechanism
                            released, mechanism = dp_copula.release_integer_vector(values, epsilon, sensitivity)
                            return released
                        cox = dp_cox.measure_cox_update(clean_private, public_model,
                                  epsilon=Fraction(plan["cox_epsilon_exact"]), public_n=dp_copula.PRIVATE_N,
                                  noise_integer_vector=noise)
                    model = {"format": "S_Copula_combined_DP_model_v1", "freeze_sha256": freeze["freeze_sha256"],
                             "accounting": plan, "copula": copula, "public_cox": public_model,
                             "cox": cox, "cox_mechanism": mechanism}
                    _atomic_json(model_path, model)
                    _append_event(ledger_dir, {"event": "model_saved", "epsilon_exact": str(eps),
                                               "model_sha256": _sha(model_path)})
                except BaseException:
                    # No private exception text/value is copied into artifacts.
                    _append_event(ledger_dir, {"event": "private_stage_failed_no_retry", "epsilon_exact": str(eps)})
                    raise
            synthetic, trace = _generate(model, item)
            temporary = csv_path.with_name("C.csv.writing")
            synthetic.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
            os.replace(temporary, csv_path)
            manifest = {"format": "S_Copula_release_manifest_v1", "epsilon_exact": str(eps), "delta": 0,
                        "total_authorized_epsilon_exact": "14", "public_n": dp_copula.PRIVATE_N,
                        "accounting": plan, "postprocess_seed": POSTPROCESS_SEED,
                        "freeze_sha256": freeze["freeze_sha256"], "model_sha256": _sha(model_path),
                        "C_sha256": _sha(csv_path), "postprocessing": trace,
                        "selection_provenance": "configuration_frozen_before_any_private_release",
                        "private_evaluation_used_for_selection": False,
                        "private_input_hash_included": False, "contains_token": False}
            _atomic_json(manifest_path, manifest)
            _append_event(ledger_dir, {"event": "complete", "epsilon_exact": str(eps), "C_sha256": manifest["C_sha256"]})
            reports.append({"epsilon": eps, "status": "complete", "directory": str(folder)})
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=HERE.parents[1] / "participant_data")
    parser.add_argument("--output-dir", type=Path, default=HERE / "releases")
    parser.add_argument("--epsilon", nargs="+", type=int, choices=EPSILONS, default=list(EPSILONS))
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.config, args.data_dir, args.output_dir, selected=tuple(args.epsilon)), ensure_ascii=False, indent=2))
    except ReleaseBlocked as exc:
        raise SystemExit(f"BLOCKED: {exc}") from None
    except Exception as exc:
        # Do not emit exception details potentially carrying private records.
        raise SystemExit(f"Release stopped ({type(exc).__name__}); inspect reservations. Never retry a private fit automatically.") from None


if __name__ == "__main__":
    main()
