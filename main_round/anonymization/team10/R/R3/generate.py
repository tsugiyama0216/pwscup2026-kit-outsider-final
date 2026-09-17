"""R3 release: deterministic aggregate calibration of frozen R2 synthetic data."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd

from calibration import calibrate
from evaluation import (HERE, DATA, COLUMNS, ZIP_NAME, digest, check_environment,
                        read_training_data, evaluate_frame, evaluate_release, write_package)

STRENGTH = 0.5
CAUSES = ('onset',)
BASE_SHA256 = 'e2c6d909b62e96b3388d0e519d0e3326d77c04b611f327c737ca50eee927570d'
SOURCE_FILES = ('generate.py', 'calibration.py', 'evaluation.py')


def load_base():
    path = HERE.parent / 'R2/C.csv'
    if digest(path) != BASE_SHA256:
        raise RuntimeError('Frozen R2 CSV changed')
    c = pd.read_csv(path, float_precision='round_trip')
    if list(c.columns) != COLUMNS:
        raise RuntimeError('Unexpected R2 columns')
    return c


def provenance():
    cfg, info = check_environment()
    info.update({'source_sha256': {name:digest(HERE/name) for name in SOURCE_FILES},
                 'base_R2_C_sha256': BASE_SHA256,
                 'R2_user_reported_result_sha256': digest(HERE/'R2_user_reported_result.json')})
    return cfg,info


def synthesize(b, cfg):
    return calibrate(b, load_base(), cfg, strength=STRENGTH, causes=CAUSES)


def csv_bytes(c):
    return c.to_csv(index=False, lineterminator='\n').encode('utf-8')


def check_improvement(score, baseline, audit):
    facets = score['facets']
    if (score['utility'] <= min(baseline.values()) or facets['U_spec'] <= baseline['U_spec']
            or facets['U_rare'] < baseline['U_rare'] - 1e-12 or facets['U_valid'] < 1.0
            or facets['U_gen'] < 0.90 or any(audit['clipped_counts'].values())):
        raise RuntimeError('R3 release quality gate failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--reproduce-check', action='store_true')
    args = parser.parse_args()
    outdir = args.output.resolve()
    if not outdir.is_relative_to(HERE):
        raise ValueError('R3 output must remain inside its model directory')
    targets = [outdir/'C.csv', outdir/'report.json', outdir/'submission'/ZIP_NAME]
    if not (args.verify_only or args.reproduce_check) and any(path.exists() for path in targets):
        raise FileExistsError('Refusing to overwrite frozen R3 artifacts')
    cfg, info = provenance()
    from pwscup2026.scoring.result import json_safe
    b = read_training_data()
    load_base()  # Validate the frozen upstream input even in verification-only mode.
    if args.verify_only or args.reproduce_check:
        saved = json.loads((outdir/'report.json').read_text(encoding='utf-8'))
        for field, value in info.items():
            if saved['provenance'].get(field) != value:
                raise RuntimeError('R3 provenance changed: '+field)
        score, validation = evaluate_release(outdir,b,cfg)
        for key in ('C_sha256','ZIP_sha256'):
            if validation[key] != saved['validation'][key]:
                raise RuntimeError('Frozen R3 artifact changed: '+key)
        if score['facets'] != saved['score']['facets'] or score['utility'] != saved['score']['utility']:
            raise RuntimeError('R3 self-score changed')
        if args.reproduce_check:
            c,_ = synthesize(b,cfg)
            if csv_bytes(c) != (outdir/'C.csv').read_bytes():
                raise RuntimeError('R3 reproduction differs from frozen CSV')
        print(json.dumps({'verified':True,'reproduced':args.reproduce_check,
                          'utility':score['utility'],'facets':score['facets']}))
        return
    c, audit = synthesize(b,cfg)
    payload = csv_bytes(c)
    scored_frame = pd.read_csv(io.BytesIO(payload),float_precision='round_trip')
    score, _ = evaluate_frame(scored_frame,b,cfg)
    feedback = json.loads((HERE/'R2_user_reported_result.json').read_text(encoding='utf-8'))
    baseline = feedback['utility_facets']
    check_improvement(score,baseline,audit)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir/'C.csv').open('xb') as stream:
        stream.write(payload)
    write_package(outdir/'submission'/ZIP_NAME,payload,(DATA/'token.txt').read_bytes())
    score, validation = evaluate_release(outdir,b,cfg)
    info.update({'model':'R3','strength':STRENGTH,'causes':list(CAUSES),
                 'method':'partial covariance-metric affine Cox calibration of frozen R2',
                 'training_inputs':['participant_data/B.csv','R2/C.csv'],
                 'evaluation_only':['participant_data/B_self.csv','participant_data/A_bg.csv','participant_data/utility_ref.json'],
                 'selection':'six fixed cause/strength settings; highest local minimum-facet utility',
                 'selection_uses_local_utility':True,'score_based_seed_search':False,
                 'formal_differential_privacy':False,'official_submission_performed':False})
    result = json_safe({'provenance':info,'model_audit':audit,'validation':validation,'score':score,
                       'comparison':{'R2_user_reported':baseline,'R3_local':score['facets'],
                                     'utility_delta':score['utility']-min(baseline.values())},
                       'reference_attack_mia':None,
                       'reference_attack_status':'R3 not officially evaluated; R2 result does not transfer'})
    with (outdir/'report.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'created':str(outdir),'rows':len(c),'utility':score['utility'],
                      'facets':score['facets']},ensure_ascii=False))


if __name__ == '__main__':
    main()
