"""Local study only: preserve utility and seek higher attack-side proxy AUC.

The official private B_ref/pool truth is unavailable. This proxy is NOT the
official reference AUC and higher attack AUC is NOT better anonymization.
"""
from pathlib import Path
import sys, json, io, hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[5]
HERE = ROOT / 'main_round/local-output/r10_research'
HERE.mkdir(parents=True, exist_ok=True)
R9 = ROOT / 'main_round/anonymization/team10/R/R9'
sys.path.insert(0, str(R9))
from evaluation import check_environment, read_training_data, evaluate_frame, DATA
from aggregate_optimizer import refine

GRID = [dict(steps=180, marginal_weight=10., rare_weight=r,
             correlation_weight=300., information_weight=100.,
             movement_weight=.02, trust_radius=.25, membership_weight=10000.,
             preserve_membership=True) for r in (4., 10., 20., 40.)]

def main():
    cfg, provenance = check_environment()
    from pwscup2026.attack.mia import overlap_mia
    from pwscup2026.scoring.attack_score import score_mia
    from pwscup2026.common.submission_schema import read_scoring_csv
    b = read_training_data()
    c = pd.read_csv(R9 / 'C.csv', float_precision='round_trip')
    expected = 'ae3d363321b58743e5f7ddf9e27b97e5525f1403c587d12ae6b5368311501136'
    if hashlib.sha256((R9 / 'C.csv').read_bytes()).hexdigest() != expected:
        raise RuntimeError('Frozen R9 changed')
    a = read_scoring_csv(DATA / 'A_bg.csv')
    targets = pd.concat([b, a], ignore_index=True)
    truth = np.r_[np.ones(len(b), dtype=int), np.zeros(len(a), dtype=int)]
    detector = json.loads((DATA / 'schema.json').read_text())['rare_detector']
    def measure(frame):
        score, valid = evaluate_frame(frame, b, cfg)
        attack = score_mia(overlap_mia(frame, a, targets, cfg), truth, cfg)
        return dict(utility=score['utility'], facets=score['facets'],
                    proxy_auc=attack.breakdown['auc'],
                    proxy_tpr_at_fpr_001=attack.breakdown['tpr_at_low_fpr'],
                    validation=valid)
    with threadpool_limits(limits=1):
        base = measure(c)
        report = dict(baseline=base, grid=GRID, candidates=[],
                      proxy_definition='All own B as positives; public A_bg as negatives and KDE background. Not official B_ref or independent holdout.',
                      purpose='User explicitly requests higher attack-side AUC, despite lower anonymity.',
                      original_sha256=expected, official_kit_commit=provenance['official_kit_commit'])
        print(json.dumps(dict(baseline=base)), flush=True)
        for index, settings in enumerate(GRID):
            out, audit = refine(b, c, cfg, detector, **settings)
            payload = out.to_csv(index=False, lineterminator='\n').encode()
            out = pd.read_csv(io.BytesIO(payload), float_precision='round_trip')
            result = measure(out)
            result.update(index=index, settings=settings, audit=audit,
                          csv_sha256=hashlib.sha256(payload).hexdigest())
            result['eligible'] = result['utility'] >= base['utility'] - 1e-12 and result['proxy_auc'] > base['proxy_auc'] + 1e-6
            (HERE / f'candidate_{index}.csv').write_bytes(payload)
            report['candidates'].append(result)
            (HERE / 'study.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
            print(json.dumps({k:result[k] for k in ['index','utility','facets','proxy_auc','proxy_tpr_at_fpr_001','eligible']}), flush=True)

if __name__ == '__main__':
    main()
