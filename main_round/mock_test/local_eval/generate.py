"""Export reproducible public and private files for one local shadow case.

Run from main_round/mock_test: python -m local_eval.generate --split dev --seed 20260927 --count 3
Only the public/ directory may be supplied to an attack predictor.  The
private/ directory exists solely for local evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

import attack_workspace as workspace
from .data import FEATURES, QI, STYLES, build_case, make_release


def export_case(*, split: str, seed: int, styles: tuple[str, ...], output: Path) -> dict:
    """Write a case from the pinned team10 inputs without exposing source row IDs."""
    if split not in ('dev', 'holdout'):
        raise ValueError('split must be dev or holdout')
    if seed < 0 or seed >= 2**32:
        raise ValueError('seed must be a nonnegative 32-bit integer')
    if not styles or len(set(styles)) != len(styles) or any(s not in STYLES for s in styles):
        raise ValueError('styles must be a nonempty list of distinct supported styles')
    output = output.resolve()
    if not output.is_relative_to(workspace.MOCK_OUTPUT.resolve()):
        raise ValueError('Output must stay under main_round/mock_test/local-output')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty; previous cases are preserved')

    verified = workspace.verify_hashes()
    source = pd.read_csv(workspace.ORIGINALS / 'A_bg.csv', float_precision='round_trip')
    background = pd.read_csv(workspace.ORIGINALS / 'B.csv', float_precision='round_trip')
    schema = json.loads((workspace.ORIGINALS / 'schema.json').read_text(encoding='utf-8'))
    case = build_case(source, background, schema, split, seed)

    public = output / 'public'
    private = output / 'private'
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir()
    case.background[FEATURES].to_csv(public / 'B_reference.csv', index=False)
    case.candidates[['record_id'] + FEATURES].to_csv(public / 'mia_candidates.csv', index=False)
    case.challenge[['challenge_row_id'] + QI].to_csv(public / 'aia_challenge.csv', index=False)
    shutil.copyfile(workspace.ORIGINALS / 'schema.json', public / 'schema.json')
    for style in styles:
        c = make_release(case.target, case.background, style, seed, schema)
        c[FEATURES].to_csv(public / f'C_{style}.csv', index=False)

    pd.DataFrame({'record_id': case.candidates.record_id.to_numpy(),
                  'member': case.mia_truth, 'detected_rare': case.mia_rare}).to_csv(
                      private / 'mia_truth.csv', index=False)
    pd.DataFrame({'challenge_row_id': case.challenge.challenge_row_id.to_numpy(),
                  'member': case.aia_member, 'detected_rare': case.aia_rare,
                  'time': case.aia_time, 'onset': case.aia_onset,
                  'pair_id': case.aia_pair}).to_csv(private / 'aia_truth.csv', index=False)
    (output / 'README.txt').write_text(
        'public/ contains only the attack inputs.\n'
        'private/ contains hidden labels for local scoring; never pass it to predict().\n'
        'The source person IDs are not exported. This is a mock case, not an official score.\n',
        encoding='utf-8')
    files = {str(p.relative_to(output)).replace('\\', '/'): workspace.sha256(p)
             for p in sorted(output.rglob('*')) if p.is_file()}
    manifest = {'format': 'pwscup2026-local-shadow-v1', 'split': split, 'seed': seed,
                'styles': list(styles), 'metadata': case.metadata,
                'input_sha256': {name: workspace.sha256(workspace.ORIGINALS / name)
                                 for name in ('A_bg.csv', 'B.csv', 'schema.json')},
                'generator_sha256': {name: workspace.sha256(Path(__file__).parent / name)
                                     for name in ('data.py', 'generate.py')},
                'files_sha256': files, 'input_verification': verified,
                'official_score_estimated': False}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                          encoding='utf-8')
    return manifest


def export_suite(*, split: str, seed: int, count: int,
                 styles: tuple[str, ...], output: Path) -> dict:
    """Save repeated samples as reusable cases; sampling does not create new people."""
    if not 1 <= count <= 1000 or seed < 0 or seed + count > 2**32:
        raise ValueError('count must be 1..1000 and all seeds must be 32-bit integers')
    if split not in ('dev', 'holdout'):
        raise ValueError('split must be dev or holdout')
    if not styles or len(set(styles)) != len(styles) or any(s not in STYLES for s in styles):
        raise ValueError('styles must be a nonempty list of distinct supported styles')
    output = output.resolve()
    if not output.is_relative_to(workspace.MOCK_OUTPUT.resolve()):
        raise ValueError('Output must stay under main_round/mock_test/local-output')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty; previous cases are preserved')
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for current_seed in range(seed, seed + count):
        case_path = output / f'seed_{current_seed}'
        export_case(split=split, seed=current_seed, styles=styles, output=case_path)
        entries.append({'path': case_path.name,
                        'manifest_sha256': workspace.sha256(case_path / 'manifest.json')})
        print(f'Generated case {len(entries)}/{count}: seed={current_seed}', flush=True)
    manifest = {'format': 'pwscup2026-local-shadow-suite-v1', 'split': split,
                'seeds': list(range(seed, seed + count)), 'styles': list(styles),
                'cases': entries, 'independent_person_samples': False,
                'person_partition_previously_used': True, 'official_score_estimated': False}
    (output / 'suite.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                       encoding='utf-8')
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('dev', 'holdout'), default='dev')
    parser.add_argument('--seed', type=int, default=20260927)
    parser.add_argument('--count', type=int, default=3,
                        help='Number of sequential sampling seeds (default: 3)')
    parser.add_argument('--style', action='append', choices=STYLES,
                        help='Repeat to select styles; by default all seven are exported')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    styles = tuple(args.style or STYLES)
    suffix = '' if args.count == 1 else f'-n{args.count}'
    destination = args.output or (workspace.MOCK_OUTPUT / 'shadow-cases' /
                                   f'{args.split}-{args.seed}{suffix}')
    if args.count == 1:
        export_case(split=args.split, seed=args.seed, styles=styles, output=destination)
    else:
        export_suite(split=args.split, seed=args.seed, count=args.count,
                     styles=styles, output=destination)
    print(f"Generated {args.count} case(s), {len(styles)} style(s) each at {destination.resolve()}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
