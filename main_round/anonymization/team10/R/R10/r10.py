"""R10: utility-preserving refinement selected for higher local attack AUC.

Reuses hash-pinned R9 optimizer. Official hidden-reference AUC is unavailable.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
R9 = HERE.parent / 'R9'
sys.path.insert(0, str(R9))
from evaluation import check_environment, read_training_data, evaluate_frame, DATA
from aggregate_optimizer import refine
from model import radii

BASE_SHA = 'ae3d363321b58743e5f7ddf9e27b97e5525f1403c587d12ae6b5368311501136'
FIXED = ['age', 'sex', 'prefecture', 'smoking', 'time', 'onset', 'death']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sources():
    dependencies = ['aggregate_optimizer.py', 'evaluation.py', 'model.py',
                    'cox_calibration.py', 'copula_helpers.py', 'joint_transport.py']
    return {**{'R9/'+name: digest(R9/name) for name in dependencies},
            **{'R10/'+name: digest(HERE/name) for name in ['r10.py','model_config.json','run.ps1']}}

def inputs():
    if digest(R9/'C.csv') != BASE_SHA:
        raise RuntimeError('Frozen R9 CSV changed')
    old = json.loads((R9/'report.json').read_text(encoding='utf-8'))['provenance']['source_sha256']
    for name in ['aggregate_optimizer.py', 'evaluation.py', 'model.py', 'cox_calibration.py', 'copula_helpers.py', 'joint_transport.py']:
        if digest(R9/name) != old[name]:
            raise RuntimeError('Frozen R9 dependency changed: '+name)
    cfg, provenance = check_environment()
    b = read_training_data()
    base = pd.read_csv(R9/'C.csv', float_precision='round_trip')
    detector = json.loads((DATA/'schema.json').read_text(encoding='utf-8'))['rare_detector']
    return cfg, provenance, b, base, detector

def proxy(frame, cfg):
    from pwscup2026.attack.mia import overlap_mia
    from pwscup2026.scoring.attack_score import score_mia
    from pwscup2026.common.submission_schema import read_scoring_csv
    b, a = read_training_data(), read_scoring_csv(DATA/'A_bg.csv')
    target = pd.concat([b,a], ignore_index=True)
    truth = np.r_[np.ones(len(b),dtype=int),np.zeros(len(a),dtype=int)]
    result = score_mia(overlap_mia(frame,a,target,cfg),truth,cfg)
    return dict(auc=float(result.breakdown['auc']),
                tpr_at_fpr_001=float(result.breakdown['tpr_at_low_fpr']),
                members=len(b), controls=len(a),
                definition='All B vs all public A_bg; same A_bg is KDE background; no independent holdout.',
                official_auc=None)

def gate(score, attack, baseline):
    if not all(np.isfinite(v) for v in score['facets'].values()):
        raise RuntimeError('Nonfinite utility')
    if score['utility'] < baseline['utility'] - 1e-12:
        raise RuntimeError('Utility decreased')
    if attack['auc'] <= baseline['proxy_auc'] + 1e-6:
        raise RuntimeError('Local proxy AUC did not increase')

def fixed_checks(base, c, detector):
    pd.testing.assert_frame_equal(base[FIXED], c[FIXED], check_exact=True)
    if not np.array_equal(radii(base,detector)>=detector['band']['lo'],
                          radii(c,detector)>=detector['band']['lo']):
        raise RuntimeError('Schema-detected rare membership changed')

def serialize(frame):
    return frame.to_csv(index=False,lineterminator='\n').encode('utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--reproduce',action='store_true')
    args = parser.parse_args()
    config = json.loads((HERE/'model_config.json').read_text(encoding='utf-8'))
    experiment = json.loads((HERE/'experiment_summary.json').read_text(encoding='utf-8'))
    selected = experiment['candidates'][config['selected_index']]
    if config['settings'] != selected['settings']:
        raise RuntimeError('Settings do not match selected candidate')
    with threadpool_limits(limits=1):
        cfg, provenance, b, base, detector = inputs()
        verifying = args.verify or args.reproduce
        if not verifying and (HERE/'C.csv').exists():
            raise RuntimeError('Refusing to overwrite existing release')
        if not verifying or args.reproduce:
            generated,audit = refine(b,base,cfg,detector,**config['settings'])
            payload = serialize(generated)
            if hashlib.sha256(payload).hexdigest() != selected['csv_sha256']:
                raise RuntimeError('Candidate byte reproduction failed')
            if args.reproduce and payload != (HERE/'C.csv').read_bytes():
                raise RuntimeError('Release byte reproduction failed')
        if verifying:
            payload = (HERE/'C.csv').read_bytes()
        c = pd.read_csv(io.BytesIO(payload),float_precision='round_trip')
        if hashlib.sha256(payload).hexdigest() != selected['csv_sha256']:
            raise RuntimeError('Release CSV changed')
        fixed_checks(base,c,detector)
        score, validation = evaluate_frame(c,b,cfg)
        attack = proxy(c,cfg)
        gate(score,attack,experiment['baseline'])
        for key,value in selected['facets'].items():
            if abs(score['facets'][key]-value)>1e-12:
                raise RuntimeError('Utility did not reproduce')
        if abs(attack['auc']-selected['proxy_auc'])>1e-12:
            raise RuntimeError('Proxy AUC did not reproduce')
        provenance['source_sha256'] = sources()
        provenance['R9_C_sha256'] = BASE_SHA
        provenance['experiment_sha256'] = digest(HERE/'experiment_summary.json')
        provenance['R9_feedback_sha256'] = digest(HERE/'R9_user_reported_result.json')
        if verifying:
            stored = json.loads((HERE/'report.json').read_text(encoding='utf-8'))
            if stored['provenance'] != provenance:
                raise RuntimeError('Input/source/environment provenance changed')
        else:
            (HERE/'C.csv').write_bytes(payload)
            report = dict(model='R10', score=score, proxy_attack=attack,
                          baseline=experiment['baseline'],validation=validation,
                          provenance=provenance,settings=config['settings'],audit=audit,
                          official_R10_result=None,
                          caveat='Higher attack AUC is not improved anonymity. Official AUC and final score cannot be guaranteed.')
            (HERE/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        result = dict(valid=True, utility=score['utility'],facets=score['facets'],
                      proxy_auc=attack['auc'], official_auc=None,
                      C_sha256=digest(HERE/'C.csv'), fixed_columns_verified=FIXED,
                      rare_membership_preserved=True,validation=validation,
                      csv_byte_reproduction=bool(args.reproduce or not verifying))
        target = 'verification_reproduce.json' if args.reproduce else 'verification.json'
        (HERE/target).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps(result),flush=True)

if __name__ == '__main__':
    main()
