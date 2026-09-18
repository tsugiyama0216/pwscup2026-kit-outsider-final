"""Release and verify the R7-based calibrated copula R8 submission."""
import argparse
import io
import json
import pandas as pd
from evaluation import (HERE, DATA, ZIP_NAME, digest, check_environment, read_training_data,
                        evaluate_frame, evaluate_release, write_package)
from model import synthesize, load_base, BASE_SHA256

SOURCES=('generate.py','model.py','copula_helpers.py','outcomes.py','tail_copula.py','joint_transport.py','cox_calibration.py','evaluation.py')

def payload(c):
    return c.to_csv(index=False,lineterminator='\n').encode('utf-8')

def release_gate(score,baseline):
    if score['utility']<=min(baseline.values())+1e-9 or any(score['facets'][k]<v-1e-12 for k,v in baseline.items()):
        raise RuntimeError('R8 must improve R7 utility without regressing any facet')

def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--verify-only',action='store_true')
    group.add_argument('--reproduce-check',action='store_true')
    args=parser.parse_args()
    cfg,provenance=check_environment()
    from pwscup2026.scoring.result import json_safe
    b=read_training_data()
    load_base()
    chosen=json.loads((HERE/'experiment_summary.json').read_text(encoding='utf-8'))['selected']
    if not chosen:
        raise RuntimeError('No improved candidate selected')
    settings={k:chosen[k] for k in ('majority_strength','uncertainty_strength','tail_strength','radial_strength','outcome_steps','uncertainty_scope','calibrate_rare_rate','uncertainty_group','tail_radius_strength','empirical_tail','active_range_limits','significance_margin','range_strategy','uncertainty_shape') if k in chosen}
    feedback=json.loads((HERE/'R7_user_reported_result.json').read_text(encoding='utf-8'))
    provenance.update({'source_sha256':{n:digest(HERE/n) for n in SOURCES},
                       'R7_C_sha256':BASE_SHA256,
                       'experiment_sha256':digest(HERE/'experiment_summary.json'),
                       'feedback_sha256':digest(HERE/'R7_user_reported_result.json')})
    if args.verify_only or args.reproduce_check:
        saved=json.loads((HERE/'report.json').read_text(encoding='utf-8'))
        if provenance!=saved['provenance'] or settings!=saved['settings']:
            raise RuntimeError('Release provenance changed')
        score,validation=evaluate_release(HERE,b,cfg)
        if validation!=saved['validation'] or score['facets']!=saved['score']['facets']:
            raise RuntimeError('Release artifact or evaluation changed')
        release_gate(score,feedback['utility_facets'])
        if args.reproduce_check:
            c,_=synthesize(b,cfg,**settings)
            if payload(c)!=(HERE/'C.csv').read_bytes():
                raise RuntimeError('CSV byte reproduction failed')
        print(json.dumps({'verified':True,'reproduced':args.reproduce_check,'utility':score['utility'],'facets':score['facets']}))
        return
    if any((HERE/p).exists() for p in ('C.csv','report.json','submission/'+ZIP_NAME)):
        raise FileExistsError('Refusing to overwrite frozen release')
    c,audit=synthesize(b,cfg,**settings)
    data=payload(c)
    score,_=evaluate_frame(pd.read_csv(io.BytesIO(data),float_precision='round_trip'),b,cfg)
    release_gate(score,feedback['utility_facets'])
    (HERE/'C.csv').write_bytes(data)
    write_package(HERE/'submission'/ZIP_NAME,data,(DATA/'token.txt').read_bytes())
    score,validation=evaluate_release(HERE,b,cfg)
    result=json_safe({'model':'R8','provenance':provenance,'settings':settings,
                      'model_audit':audit,'validation':validation,'score':score,
                      'comparison':{'R7_user_reported':feedback,'R8_local':score['facets'],
                                    'utility_delta':score['utility']-min(feedback['utility_facets'].values())},
                      'training_inputs':['participant_data/B.csv','R7/C.csv','participant_data/schema.json rare_detector'],
                      'evaluation_only':['B_self.csv','A_bg.csv','utility_ref.json'],
                      'formal_differential_privacy':False,'official_attack_evaluated':False,
                      'official_submission_performed':False})
    with (HERE/'report.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'utility':score['utility'],'facets':score['facets'],'validation':validation}))

if __name__=='__main__':
    main()
