"""Export reproducible public and private files for one local shadow case.

Run from main_round/attack/attack_TS with: python -m local_eval.generate --split dev --seed 20260927
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
    if not output.is_relative_to(workspace.TS_OUTPUT.resolve()):
        raise ValueError('Output must stay under main_round/attack/attack_TS/local-output')
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('dev', 'holdout'), default='dev')
    parser.add_argument('--seed', type=int, default=20260927)
    parser.add_argument('--style', action='append', choices=STYLES,
                        help='Repeat to select styles; by default all seven are exported')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    styles = tuple(args.style or STYLES)
    destination = args.output or (workspace.TS_OUTPUT / 'shadow-cases' /
                                   f'{args.split}-{args.seed}')
    manifest = export_case(split=args.split, seed=args.seed, styles=styles,
                           output=destination)
    print(f"Generated {len(styles)} style(s) at {destination.resolve()}")
    print(f"Source files verified: {manifest['input_verification']['verified_files']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
