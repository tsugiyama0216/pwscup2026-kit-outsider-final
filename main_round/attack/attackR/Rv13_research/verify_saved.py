"""Verify frozen artifacts and published score summaries without local trial data."""
from pathlib import Path, PureWindowsPath
import csv, hashlib, json, math, sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'main_round/mock_test'))
from rank_validation.ranking_metric import compute_metrics


def verify():
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
    comparisons = read(HERE / 'comparison.json')
    summary = {r['model']: r for r in comparisons['models']}
    counts = {}
    for model in ('Rv13', 'Rv14', 'Rv15'):
        folder = HERE.parent / model
        manifest = read(folder / 'artifact_manifest.json')
        for path, expected in manifest.items():
            if sha(folder / path) != expected:
                raise ValueError(f'{model}: frozen artifact differs: {path}')
        if sha(folder / 'model.py') != read(folder / 'selection.json')['source_sha256']:
            raise ValueError(f'{model}: source differs from selection')
        counts[model] = len(manifest)
    for model in ('Rv2', 'Rv13', 'Rv14', 'Rv15'):
        row = summary[model]
        actual = read(HERE / 'evidence' / f'prelim-{model}.json')['summary']['attack_power']
        if not math.isclose(actual, row['prelim_attack_power'], rel_tol=0, abs_tol=1e-14):
            raise ValueError(f'{model}: preliminary summary differs')
        for split, seeds in [('development', [2026100961, 2026100962]), ('confirmation', [2026100981, 2026100982])]:
            values = []
            for seed in seeds:
                with (HERE / 'evidence' / f'{seed}-{model}-by_target.csv').open(newline='', encoding='utf8') as source:
                    values.append(compute_metrics(csv.DictReader(source))['stable_index'])
            if not math.isclose(math.fsum(values) / 2, row[split + '_stable_index'], rel_tol=0, abs_tol=1e-14):
                raise ValueError(f'{model}: {split} summary differs')
        if row['both_requested_scores_above_Rv2']:
            raise ValueError('Unexpected claim that both requested scores exceed Rv2')
    for source, expected in comparisons['evidence_sha256'].items():
        parts = PureWindowsPath(source).parts
        if 'panels' in parts:
            i = parts.index('panels')
            destination = HERE / 'evidence' / f'{parts[i+1]}-{parts[i+3]}-{parts[-1]}'
        else:
            model = PureWindowsPath(source).stem if parts[-2] == 'evaluations' else parts[-2]
            destination = HERE / 'evidence' / f'prelim-{model}.json'
        if sha(destination) != expected:
            raise ValueError('Copied score evidence differs: ' + destination.name)
    return {'frozen_artifact_counts': counts, 'score_evidence_and_summaries_match': True, 'both_requested_scores_exceed_Rv2': False}


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
