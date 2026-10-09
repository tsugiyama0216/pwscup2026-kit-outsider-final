"""Public-distribution coverage gate for extrapolating local model comparisons.

This module never reads feedback, attack predictions, B/B_self values, membership
truth or AIA truth. It does not estimate anonymization strength or official scores.
The two gate metrics were defined from public inputs before feedback comparison.
Run: python -m rank_validation.public_profiles --root REPO --output NEW_DIRECTORY
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

QI = ('age', 'SBP', 'TG', 'HDL', 'ALT')
CLINICAL = ('BMI', 'SBP', 'TG', 'HDL', 'ALT', 'FPG')
CONTINUOUS = ('age',) + CLINICAL
CATEGORIES = ('sex', 'prefecture')
GATE_METRICS = ('challenge_qi_nn_median', 'challenge_to_background_nn_ratio')
LOCAL_R_VERSIONS = (1, 3, 4, 5, 6, 7, 9, 10, 11, 12)
LIMITATIONS = [
    'Passing public range checks does not establish correct model ranking.',
    'The AIA challenge mixes hidden member and control rows. Public distances do not identify those groups or measure low-FPR/rare attack performance.',
    'Cohort and background shifts confound cross-round comparisons. Distances are not an anonymization-strength scale.',
    'The local twelve defenses originate from one cohort and include ten related R versions; they are not twelve independent defense families.',
    'Metric ranges are marginal envelopes, not joint-density coverage or a calibrated probability.',
    'A coverage failure rejects extrapolation; it neither reverses a model ranking nor predicts which model is better.',
]


def _sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _finite(value) -> bool:
    try:
        return value is not None and bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _median(values) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def verify_original_inputs(root: Path) -> dict:
    """Hash the existing manifests before parsing public tables; no token read."""
    root = Path(root).resolve()
    attack = root / 'main_round/attack'
    prelim = root / 'main_round/mock_test/mock_TS_using_prelim'
    main_path = attack / 'source_manifest.json'
    prelim_path = prelim / 'provenance/data_manifest.json'
    main_entries = _json(main_path)['files']
    prelim_entries = [item for item in _json(prelim_path)['files']
                      if item['path'].startswith('data/public/')]
    for label, base, boundary, entries in (
        ('main', attack, root / 'main_round', main_entries),
        ('preliminary public', prelim, prelim, prelim_entries),
    ):
        _require(bool(entries), f'Empty {label} manifest')
        for item in entries:
            path = (base / item['path']).resolve()
            _require(not Path(item['path']).is_absolute() and path.is_relative_to(boundary),
                     f'{label} manifest path escapes allowed tree')
            _require(path.name.lower() != 'token.txt', 'Credential path in manifest')
            _require(path.is_file() and _sha(path) == item['sha256'],
                     f'{label} original hash mismatch: {item["path"]}')
    return {
        'main_original_manifest_verified': len(main_entries),
        'preliminary_public_entries_verified': len(prelim_entries),
        'main_manifest_sha256': _sha(main_path),
        'preliminary_manifest_sha256': _sha(prelim_path),
        'hash_checks_only_for_B_and_B_self': True,
    }


def _validated_table(frame: pd.DataFrame, columns: Iterable[str], label: str) -> pd.DataFrame:
    _require(len(frame) > 0, f'{label} is empty')
    columns = list(columns)
    _require(set(columns + list(CATEGORIES)).issubset(frame.columns), f'{label}: missing columns')
    _require(not frame[list(CATEGORIES)].isna().any().any(), f'{label}: missing categories')
    _require(np.isfinite(frame[columns].to_numpy(float)).all(), f'{label}: nonfinite values')
    return frame.reset_index(drop=True)


def nearest_distances(query: pd.DataFrame, c: pd.DataFrame,
                      columns: Iterable[str], background: pd.DataFrame) -> np.ndarray:
    """Background-SD RMS distance, exact sex/prefecture; missing strata are NaN."""
    columns = list(columns)
    query = query.reset_index(drop=True)
    scale = background[columns].std(ddof=0).to_numpy(float)
    scale[scale == 0] = 1.0
    distances = np.full(len(query), np.nan)
    for key, indices in query.groupby(list(CATEGORIES), sort=False).groups.items():
        mask = (c.sex == key[0]) & (c.prefecture == key[1])
        if not mask.any():
            continue
        tree = cKDTree(c.loc[mask, columns].to_numpy(float) / scale)
        distances[np.asarray(list(indices), int)] = tree.query(
            query.loc[indices, columns].to_numpy(float) / scale)[0] / np.sqrt(len(columns))
    return distances


def profile_tables(c: pd.DataFrame, challenge: pd.DataFrame, background: pd.DataFrame) -> dict:
    """Profile public tables with fixed, interpretable metrics; no hidden labels."""
    c = _validated_table(c, CONTINUOUS + ('time', 'onset', 'death'), 'C')
    challenge = _validated_table(challenge, QI, 'challenge')
    background = _validated_table(background, CONTINUOUS, 'background')
    challenge_dist = nearest_distances(challenge, c, QI, background)
    background_dist = nearest_distances(background, c, QI, background)
    full_background_dist = nearest_distances(background, c, CONTINUOUS, background)
    cv = c[list(CLINICAL)].to_numpy(float)
    upper = np.triu_indices(len(CONTINUOUS), 1)
    correlation_diff = np.abs(c[list(CONTINUOUS)].corr().to_numpy()[upper]
                              - background[list(CONTINUOUS)].corr().to_numpy()[upper])
    finite_corr = correlation_diff[np.isfinite(correlation_diff)]
    scale = background[list(CONTINUOUS)].std(ddof=0)
    challenge_median = _median(challenge_dist)
    background_median = _median(background_dist)
    ratio = (challenge_median / max(background_median, 1e-12)
             if challenge_median is not None and background_median is not None else None)
    return {
        'n': len(c), 'n_challenge': len(challenge),
        'clinical_one_decimal_fraction': float(np.isclose(cv * 10, np.round(cv * 10), atol=1e-7, rtol=0).mean()),
        'clinical_integer_fraction': float(np.isclose(cv, np.round(cv), atol=1e-8, rtol=0).mean()),
        'clinical_unique_fraction_mean': float(np.mean([c[col].nunique() / len(c) for col in CLINICAL])),
        'duplicate_feature_fraction': float(c.drop(columns=['record_id'], errors='ignore').duplicated().mean()),
        'correlation_abs_difference_bg': float(finite_corr.mean()) if len(finite_corr) else None,
        'wasserstein_sd_mean': float(np.mean([
            wasserstein_distance(c[col], background[col]) / (scale[col] or 1.0) for col in CONTINUOUS])),
        'time_10_fraction': float(np.isclose(c.time, 10, atol=1e-9, rtol=0).mean()),
        'time_unique_fraction': float(c.time.nunique() / len(c)),
        'time_ge_9p5_fraction': float((c.time >= 9.5).mean()),
        'time_mode_value': float(c.time.mode().iloc[0]),
        'time_mode_fraction': float(c.time.value_counts(normalize=True).iloc[0]),
        'onset_fraction': float(c.onset.mean()), 'death_fraction': float(c.death.mean()),
        'time_median': float(c.time.median()), 'time_q10': float(c.time.quantile(.1)),
        'challenge_qi_exact_fraction': float((np.nan_to_num(challenge_dist, nan=np.inf) <= 1e-10).mean()),
        'challenge_missing_stratum_fraction': float((~np.isfinite(challenge_dist)).mean()),
        'challenge_qi_nn_median': challenge_median,
        'background_qi_nn_median': background_median,
        'background_full_nn_median': _median(full_background_dist),
        'challenge_to_background_nn_ratio': ratio,
    }


def build_coverage_report(training_profiles: Iterable[dict], target_profiles: Iterable[dict]) -> dict:
    """Gate extrapolation using the two frozen marginal public metric ranges.

    Any target outside either range, or missing a metric, marks the population
    comparison unsupported. An in-range result still does not validate ranking.
    No target feedback is used to set bounds or to select gate metrics.
    """
    training, targets = list(training_profiles), list(target_profiles)
    _require(bool(training) and bool(targets), 'Coverage requires nonempty training and target profiles')
    envelopes = {}
    for metric in GATE_METRICS:
        values = [float(row[metric]) for row in training if _finite(row.get(metric))]
        envelopes[metric] = {
            'training_min': min(values) if values else None,
            'training_max': max(values) if values else None,
            'training_complete': len(values) == len(training),
            'outside_count': 0, 'unknown_count': 0,
            'target_count': len(targets),
        }
    target_results = []
    for row in targets:
        outside, unknown = [], []
        for metric, envelope in envelopes.items():
            if not envelope['training_complete'] or not _finite(row.get(metric)):
                unknown.append(metric)
                envelope['unknown_count'] += 1
                continue
            value = float(row[metric])
            lo, hi = envelope['training_min'], envelope['training_max']
            if ((value < lo and not np.isclose(value, lo, rtol=1e-12, atol=1e-12))
                    or (value > hi and not np.isclose(value, hi, rtol=1e-12, atol=1e-12))):
                outside.append(metric)
                envelope['outside_count'] += 1
        target_results.append({
            'name': str(row.get('name', 'unnamed')),
            'cohort': int(row['cohort']) if _finite(row.get('cohort')) else None,
            'unsupported': bool(outside or unknown),
            'outside_metrics': outside, 'unknown_metrics': unknown,
        })
    unsupported_count = sum(row['unsupported'] for row in target_results)
    unsupported = unsupported_count > 0
    return {
        'schema_version': 1,
        'training_count': len(training), 'target_count': len(targets),
        'gate_metrics': list(GATE_METRICS), 'metric_ranges': envelopes,
        'targets': target_results, 'unsupported_target_count': unsupported_count,
        'model_ranking': {
            'unsupported': unsupported,
            'status': ('unsupported_for_target_population' if unsupported
                       else 'coverage_pass_only_rank_validity_unproven'),
            'reason': ('At least one target lies outside the observed local public ranges or has unknown coverage.'
                       if unsupported else 'Public ranges cover these targets, but ranking validity remains unproven.'),
            'official_rank_validated': False,
        },
        'is_anonymization_strength_score': False,
        'uses_official_feedback': False,
        'limitations': list(LIMITATIONS),
    }


def apply_rank_scope(rankings: Iterable[dict], coverage_report: dict) -> list[dict]:
    """Attach the domain restriction without changing any scores or ordering."""
    scope = coverage_report['model_ranking']
    return [dict(row, ranking_scope=dict(scope)) for row in rankings]


def run(root: Path, output: Path) -> dict:
    root, output = Path(root).resolve(), Path(output).resolve()
    output_base = root / 'main_round/mock_test/local-output'
    _require(output.is_relative_to(output_base) and output != output_base,
             'Output must be a new directory below main_round/mock_test/local-output')
    _require(not output.exists(), 'Output already exists; choose a new directory')
    verification = verify_original_inputs(root)
    output.mkdir(parents=True, exist_ok=False)
    attack = root / 'main_round/attack'
    prelim = root / 'main_round/mock_test/mock_TS_using_prelim'
    input_hashes = {}

    def read(path):
        input_hashes[path.relative_to(root).as_posix()] = _sha(path)
        return pd.read_csv(path, float_precision='round_trip')

    backgrounds = {
        'main': read(root / 'main_round/anonymization/team10/participant_data/A_bg.csv'),
        'prelim': read(prelim / 'data/public/team24_processing/A_bg.csv'),
    }
    rows = []

    def add(name, cohort, group, phase, c_path, challenge_path):
        row = {'name': name, 'cohort': cohort, 'group': group,
               'c_path': c_path.relative_to(root).as_posix(),
               'challenge_path': challenge_path.relative_to(root).as_posix()}
        row.update(profile_tables(read(c_path), read(challenge_path), backgrounds[phase]))
        rows.append(row)

    for phase, directory, own in (
        ('main', attack / 'targets', 10),
        ('prelim', prelim / 'data/public/attack_targets', 24),
    ):
        for c_path in sorted(directory.glob('C_*.csv'), key=lambda p: int(p.stem[2:])):
            cohort = int(c_path.stem[2:])
            if cohort != own:
                add(f'C_{cohort}', cohort, phase, phase, c_path,
                    directory / f'aia_challenge_{cohort}.csv')
    base = root / 'main_round/anonymization/team10'
    releases = [('M1_v5', base / 'M/M1_v5/C.csv'), ('S1', base / 'S/S1/C.csv')]
    releases.extend((f'R{k}', base / f'R/R{k}/C.csv') for k in LOCAL_R_VERSIONS)
    for name, path in releases:
        add(name, 10, 'local_12', 'main', path, attack / 'targets/aia_challenge_10.csv')
    groups = {label: [row for row in rows if row['group'] == label]
              for label in ('main', 'prelim', 'local_12')}
    _require({key: len(value) for key, value in groups.items()} == {'main': 22, 'prelim': 20, 'local_12': 12},
             'Unexpected cohort inventory for the pinned 2026 snapshots')
    coverage = build_coverage_report(groups['local_12'], groups['main'])
    preliminary_coverage = build_coverage_report(groups['local_12'], groups['prelim'])
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'profiles.csv', index=False)
    _dump(output / 'profiles.json', rows)
    _dump(output / 'coverage.json', coverage)
    _dump(output / 'preliminary_coverage.json', preliminary_coverage)
    _dump(output / 'input_verification.json', verification)
    _dump(output / 'inputs.json', {
        'files': input_hashes, 'code_sha256': _sha(Path(__file__)),
        'uses_official_feedback': False, 'uses_private_truth': False,
        'uses_B_or_B_self_values': False,
        'distance_definition': 'Nearest C row within exact sex/prefecture, RMS of age/SBP/TG/HDL/ALT differences in round-specific A_bg standard-deviation units.',
        'ratio_definition': 'Median challenge-to-C distance divided by median A_bg-to-C distance.',
        'time_note': 'Exact time=10 differs from approximately 9.9986; time>=9.5 and modal-time diagnostics avoid interpreting that representation difference as stronger anonymization.',
    })
    summary = []
    for group, group_frame in frame.groupby('group'):
        for metric in frame.select_dtypes(include='number').columns:
            if metric in ('cohort', 'n', 'n_challenge'):
                continue
            series = group_frame[metric].dropna()
            summary.append({'group': group, 'metric': metric, 'count': len(series),
                            'min': float(series.min()) if len(series) else None,
                            'median': float(series.median()) if len(series) else None,
                            'max': float(series.max()) if len(series) else None})
    pd.DataFrame(summary).to_csv(output / 'summary.csv', index=False)
    return {'output': str(output), 'verification': verification,
            'counts': {key: len(value) for key, value in groups.items()},
            'coverage': coverage}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.root, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != 'coverage'}, ensure_ascii=False))
    print(json.dumps(result['coverage']['model_ranking'], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
