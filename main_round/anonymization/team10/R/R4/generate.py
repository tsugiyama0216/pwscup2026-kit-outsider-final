"""R4 release: deterministic aggregate calibration of frozen R3 synthetic data."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd

from joint_transport import transport
from evaluation import (HERE, DATA, COLUMNS, ZIP_NAME, digest, check_environment,
                        read_training_data, evaluate_frame, evaluate_release, write_package)

STRENGTH = 0.5
PRESERVE_RISK = True
BASE_SHA256 = '483c2ffce37af5a66d72a73db7ada09c1a2b0b571b8d3a17966dfc529d20f54b'
SOURCE_FILES = ('generate.py', 'joint_transport.py', 'cox_calibration.py', 'evaluation.py')


def load_base():
    path = HERE.parent / 'R3/C.csv'
    if digest(path) != BASE_SHA256:
        raise RuntimeError('Frozen R3 CSV changed')
    c = pd.read_csv(path, float_precision='round_trip')
    if list(c.columns) != COLUMNS:
        raise RuntimeError('Unexpected R3 columns')
    return c


def provenance():
    cfg, info = check_environment()
    info.update({'source_sha256': {name:digest(HERE/name) for name in SOURCE_FILES},
                 'base_R3_C_sha256': BASE_SHA256,
                 'R3_user_reported_result_sha256': digest(HERE/'R3_user_reported_result.json')})
    return cfg,info


def synthesize(b, cfg):
    return transport(b, load_base(), cfg, strength=STRENGTH, cox_strength=0.0, preserve_risk=PRESERVE_RISK)


def csv_bytes(c):
    return c.to_csv(index=False, lineterminator='\n').encode('utf-8')


def check_improvement(score, baseline, audit):
    facets = score['facets']
    if (score['utility'] <= min(baseline.values())
            or any(facets[name] < value - 1e-12 for name,value in baseline.items())
            or any(audit['clipped_counts'].values())
            or audit['risk_projection']['maximum_log_risk_change_before_clipping'] > 1e-10):
        raise RuntimeError('R4 release must improve utility without regressing any R3 facet or Cox risk')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--reproduce-check', action='store_true')
    args = parser.parse_args()
    outdir = args.output.resolve()
    if not outdir.is_relative_to(HERE):
        raise ValueError('R4 output must remain inside its model directory')
    targets = [outdir/'C.csv', outdir/'report.json', outdir/'submission'/ZIP_NAME]
    if not (args.verify_only or args.reproduce_check) and any(path.exists() for path in targets):
        raise FileExistsError('Refusing to overwrite frozen R4 artifacts')
    cfg, info = provenance()
    from pwscup2026.scoring.result import json_safe
    b = read_training_data()
    load_base()  # Validate the frozen upstream input even in verification-only mode.
    if args.verify_only or args.reproduce_check:
        saved = json.loads((outdir/'report.json').read_text(encoding='utf-8'))
        for field, value in info.items():
            if saved['provenance'].get(field) != value:
                raise RuntimeError('R4 provenance changed: '+field)
        score, validation = evaluate_release(outdir,b,cfg)
        for key in ('C_sha256','ZIP_sha256'):
            if validation[key] != saved['validation'][key]:
                raise RuntimeError('Frozen R4 artifact changed: '+key)
        if score['facets'] != saved['score']['facets'] or score['utility'] != saved['score']['utility']:
            raise RuntimeError('R4 self-score changed')
        if args.reproduce_check:
            c,_ = synthesize(b,cfg)
            if csv_bytes(c) != (outdir/'C.csv').read_bytes():
                raise RuntimeError('R4 reproduction differs from frozen CSV')
        print(json.dumps({'verified':True,'reproduced':args.reproduce_check,
                          'utility':score['utility'],'facets':score['facets']}))
        return
    c, audit = synthesize(b,cfg)
    payload = csv_bytes(c)
    scored_frame = pd.read_csv(io.BytesIO(payload),float_precision='round_trip')
    score, _ = evaluate_frame(scored_frame,b,cfg)
    feedback = json.loads((HERE/'R3_user_reported_result.json').read_text(encoding='utf-8'))
    baseline = feedback['utility_facets']
    check_improvement(score,baseline,audit)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir/'C.csv').open('xb') as stream:
        stream.write(payload)
    write_package(outdir/'submission'/ZIP_NAME,payload,(DATA/'token.txt').read_bytes())
    score, validation = evaluate_release(outdir,b,cfg)
    info.update({'model':'R4','strength':STRENGTH,'preserve_risk':PRESERVE_RISK,
                 'method':'conditional joint-distribution transport preserving frozen R3 Cox risk',
                 'training_inputs':['participant_data/B.csv','R3/C.csv'],
                 'evaluation_only':['participant_data/B_self.csv','participant_data/A_bg.csv','participant_data/utility_ref.json'],
                 'selection':'22 settings in 3 algorithm batches and a local refinement; highest utility, then Pareto-dominating facets on ties',
                 'selection_uses_local_utility':True,'score_based_seed_search':False,
                 'formal_differential_privacy':False,'official_submission_performed':False})
    result = json_safe({'provenance':info,'model_audit':audit,'validation':validation,'score':score,
                       'comparison':{'R3_user_reported':baseline,'R4_local':score['facets'],
                                     'utility_delta':score['utility']-min(baseline.values())},
                       'reference_attack_mia':None,
                       'reference_attack_status':'R4 not officially evaluated; R3 result does not transfer'})
    with (outdir/'report.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'created':str(outdir),'rows':len(c),'utility':score['utility'],
                      'facets':score['facets']},ensure_ascii=False))


if __name__ == '__main__':
    main()
