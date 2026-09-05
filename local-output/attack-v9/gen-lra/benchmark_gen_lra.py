"""New-method feasibility benchmark with exact self-challenge membership.

Discovery uses controlled modifications of own B. The actual public C24 is
evaluated on different sex/age query blocks after configuration selection.
Only memberships are known for controls: do not claim a measured AIA time score.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from attack_phase_v8.actual_inputs import load_actual_inputs
from attack_phase_v9.gen_lra import LocalLikelihoodInfluence, reference_bandwidth
from attack_phase_daybreak_independent.generate_submission_daybreak_independent import RobustAtomEncoder
from attack_phase_v5.aia_structured_ensemble import age_band
from pwscup2026.scoring.attack_score import _tpr_at_fpr

OUT = Path(__file__).resolve().parent
DIST = Path('C:/Users/takut/Downloads/PWSCup2026_prelim_team24_アウトサイダー_outsider')
TARGETS = ROOT / 'team-shared/prelim/attack-targets'
QI = ['age', 'SBP', 'TG', 'HDL', 'ALT', 'sex', 'prefecture']
VIEWS = {'stable3_sex': ['age', 'SBP', 'TG', 'sex'],
         'qi5_sex': ['age', 'SBP', 'TG', 'HDL', 'ALT', 'sex']}


def metrics(y, scores):
    tpr, auc = _tpr_at_fpr(np.asarray(scores), np.asarray(y), .01)
    return {'tpr01': tpr, 'auc': auc}


def all_scores(query, release, reference, view):
    results = {}
    for multiplier in (.5, 1., 2.):
        h = reference_bandwidth(reference, multiplier=multiplier)
        family = LocalLikelihoodInfluence(reference, h).score_many(query, release)
        for name, values in family.items():
            label = f'{view}:h{multiplier}:{name}' if name != 'dcr' else f'{view}:dcr'
            results[label] = values
    return results


def main():
    actual = load_actual_inputs(DIST, TARGETS)
    challenge = actual.self_challenge.reset_index(drop=True)
    y = pd.MultiIndex.from_frame(challenge[QI]).isin(
        pd.MultiIndex.from_frame(actual.candidates[QI])).astype(int)
    assert len(y) == 326 and y.sum() == 163
    groups = challenge.sex.astype(str).to_numpy() + ':' + age_band(challenge.age).astype(str)
    # Whole public match blocks remain on one side; this uses no outcome label.
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=91005).split(challenge, groups=groups))
    assert not set(groups[train]) & set(groups[test])
    home = str(challenge.prefecture.iloc[0])
    background = actual.background.loc[actual.background.prefecture.astype(str).eq(home)].reset_index(drop=True)
    source = actual.candidates.loc[actual.candidates.prefecture.astype(str).eq(home)].reset_index(drop=True)
    c24 = pd.read_csv(TARGETS / 'C_24.csv')
    c24 = c24.loc[c24.prefecture.astype(str).eq(home)].reset_index(drop=True)
    encoder = RobustAtomEncoder.fit(actual.background)
    encoded = {view: (encoder.matrix(challenge, columns), encoder.matrix(source, columns),
                      encoder.matrix(background, columns), encoder.matrix(c24, columns))
               for view, columns in VIEWS.items()}
    discovery_rows = []
    for seed in (91011, 91012):
        for mechanism, noise in (('jitter', .04), ('jitter', .12), ('bootstrap', .08)):
            for view, (query, b, ref, _released) in encoded.items():
                rng = np.random.default_rng(seed)
                indices = np.arange(len(b)) if mechanism == 'jitter' else rng.choice(len(b), len(b), replace=True)
                released = b[indices].copy()
                # Last dimension is sex and is not noised.
                released[:, :-1] += rng.normal(0, noise, size=released[:, :-1].shape)
                family = all_scores(query[train], released, ref, view)
                for name, values in family.items():
                    discovery_rows.append({'seed': seed, 'mechanism': mechanism, 'noise': noise,
                                           'name': name, **metrics(y[train], values)})
            pd.DataFrame(discovery_rows).to_csv(OUT / 'discovery_scores.csv', index=False)
            print(f'discovery checkpoint: {seed} {mechanism} {noise}', flush=True)
    discovery = pd.DataFrame(discovery_rows)
    summary = discovery.groupby('name')[['tpr01', 'auc']].mean().reset_index()
    ordered = summary.sort_values(['tpr01', 'auc', 'name'], ascending=[False, False, True])
    selected = ordered.loc[ordered.name.str.contains('gen_lra')].iloc[0]['name']
    baseline = ordered.loc[~ordered.name.str.contains('gen_lra')].iloc[0]['name']
    summary.to_csv(OUT / 'discovery_summary.csv', index=False)
    (OUT / 'locked_selection.json').write_text(json.dumps({'method': selected, 'baseline': baseline,
        'rule': 'best mean TPR01; AUC then name tie-break, discovery only'}, indent=2), encoding='utf-8')
    print(f'locked: {selected} vs {baseline}', flush=True)

    actual_scores = {}
    null_scores = {}
    for view, (query, _b, ref, released) in encoded.items():
        actual_scores.update(all_scores(query, released, ref, view))
        # Negative control deliberately removes the actual C24 release.
        null_release = ref[np.random.default_rng(91099).choice(len(ref), len(released), replace=True)]
        null_scores.update(all_scores(query, null_release, ref, view))
    diagnostic = pd.DataFrame([{'name': name, **metrics(y[test], values[test])}
                              for name, values in actual_scores.items()])
    diagnostic.to_csv(OUT / 'actual_heldout_diagnostics_not_for_selection.csv', index=False)
    method_metrics = metrics(y[test], actual_scores[selected][test])
    baseline_metrics = metrics(y[test], actual_scores[baseline][test])
    delta = {key: method_metrics[key] - baseline_metrics[key] for key in method_metrics}
    # Cluster bootstrap over test match blocks: conditional uncertainty in this
    # one reused cohort, not uncertainty over other anonymization mechanisms.
    rng = np.random.default_rng(91101)
    blocks = [test[groups[test] == group] for group in np.unique(groups[test])]
    bootstrap = []
    for _ in range(499):
        indices = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        m = metrics(y[indices], actual_scores[selected][indices])
        b = metrics(y[indices], actual_scores[baseline][indices])
        bootstrap.append({key: m[key] - b[key] for key in m})
    interval = {key: np.quantile([r[key] for r in bootstrap], [.05, .95]).tolist() for key in delta}
    null = metrics(y[test], null_scores[selected][test])
    gate = bool(method_metrics['auc'] >= .60 and delta['auc'] >= .03
                and interval['auc'][0] > 0 and delta['tpr01'] >= 0
                and method_metrics['auc'] > null['auc'] + .03)
    report = {'status': 'feasibility_pass_requires_target_validation' if gate else 'REJECT',
        'selected': selected, 'selected_baseline': baseline,
        'actual_test': method_metrics, 'baseline_actual_test': baseline_metrics,
        'delta': delta, 'paired_block_bootstrap_90pct': interval,
        'background_release_negative_control': null, 'gate_pass': gate,
        'train_rows': len(train), 'test_rows': len(test), 'train_blocks': len(set(groups[train])),
        'test_blocks': len(set(groups[test])), 'members_train': int(y[train].sum()),
        'members_test': int(y[test].sum()), 'reference_home_rows': len(background),
        'source_home_rows': len(source), 'release_home_rows': len(c24),
        'input_provenance': actual.provenance,
        'limitations': ['Membership classification only; hidden control times are unknown, so this is not AIA power.',
            'One self target, previously used in other research; method selection uses disjoint query blocks but cannot certify cross-target transfer.',
            'Discovery modifications are assumed mechanisms; the actual C24 validation mechanism is not assumed.',
            'Comparison baseline is KDE/DCR, not the full v7 attack; pass would not authorize automatic promotion.',
            'Nonselected heldout diagnostics are recorded for audit only, not retuning.'],
        'production_changed': False, 'token_read': False, 'submission_performed': False}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    pd.DataFrame({'challenge_row_id': challenge.challenge_row_id, 'known_member': y,
                  'heldout': np.isin(np.arange(len(y)), test), 'selected': actual_scores[selected],
                  'baseline': actual_scores[baseline]}).to_csv(OUT / 'query_scores.csv', index=False)
    print(json.dumps({key: report[key] for key in ['status', 'selected', 'selected_baseline', 'actual_test',
        'baseline_actual_test', 'delta', 'paired_block_bootstrap_90pct', 'background_release_negative_control']}), flush=True)


if __name__ == '__main__':
    main()
