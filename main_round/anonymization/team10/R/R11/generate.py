"""Generate, package and verify the selected R11 without changing earlier models."""
import argparse,hashlib,io,json,zipfile
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from refinement import (HERE,R9,DATA,BASE_SHA,LABS,load_base,synthesize,radii,
                        check_environment,read_training_data,evaluate_frame,digest)

ZIP_NAME='team10_R11_defense.zip'
SOURCES=['generate.py','refinement.py','optimizer.py','research.py','research_joint.py','research_locked.py','select_release.py','run.ps1']
DEPENDENCIES=['evaluation.py','model.py','aggregate_optimizer.py','cox_calibration.py','copula_helpers.py','joint_transport.py']

def payload(frame):
    return frame.to_csv(index=False,lineterminator='\n').encode('utf-8')

def release_gate(score,baseline):
    if not np.isfinite(score['utility']) or score['utility']<=min(baseline.values())+1e-9:
        raise RuntimeError('R11 must improve R9 utility')
    if any(not np.isfinite(score['facets'][k]) or score['facets'][k]<v-1e-12 for k,v in baseline.items()):
        raise RuntimeError('A utility facet regressed from R9')

def verify_fixed(base,c):
    pd.testing.assert_frame_equal(base.drop(columns=LABS),c.drop(columns=LABS),check_exact=True)
    detector=json.loads((DATA/'schema.json').read_text(encoding='utf-8'))['rare_detector']
    np.testing.assert_array_equal(radii(base,detector)>=detector['band']['lo'],radii(c,detector)>=detector['band']['lo'])

def package(data):
    path=HERE/'submission'/ZIP_NAME
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,content in [('C.csv',data),('token.txt',(DATA/'token.txt').read_bytes())]:
            info=zipfile.ZipInfo(name,date_time=(2026,9,19,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED; info.create_system=3; info.external_attr=0o600<<16
            z.writestr(info,content)

def validate_package(b,cfg):
    from pwscup2026.kit.validate import check_package,validate_submission_zip
    from pwscup2026.kit.selfscore import load_submitted_c
    path=HERE/'submission'/ZIP_NAME
    check=check_package(path,cfg)
    validated=validate_submission_zip(path,cfg,dist_dir=DATA,kind_override='defense')
    if not check.ok or not validated.ok or check.kind!='defense': raise RuntimeError('Invalid official package')
    with zipfile.ZipFile(path) as z:
        if z.namelist()!=['C.csv','token.txt'] or z.testzip() is not None: raise RuntimeError('Invalid ZIP structure/CRC')
        if z.read('C.csv')!=(HERE/'C.csv').read_bytes(): raise RuntimeError('CSV mismatch')
        if z.read('token.txt')!=(DATA/'token.txt').read_bytes(): raise RuntimeError('Token mismatch')
    c=load_submitted_c(path)
    verify_fixed(load_base(),c)
    score,validation=evaluate_frame(c,b,cfg)
    validation.update(package_ok=True,flat_zip_entries=['C.csv','token.txt'],csv_matches_zip=True,
                      token_matches_original=True,fixed_columns_preserved=True,rare_membership_preserved=True,
                      C_sha256=digest(HERE/'C.csv'),ZIP_sha256=digest(path),ZIP_bytes=path.stat().st_size)
    return score,validation

def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--verify-only',action='store_true')
    group.add_argument('--reproduce-check',action='store_true')
    args=parser.parse_args()
    cfg,provenance=check_environment(); b=read_training_data(); load_base()
    selection=json.loads((HERE/'experiment_summary.json').read_text(encoding='utf-8'))
    chosen=selection['selected']; settings=chosen['settings']
    feedback=json.loads((HERE/'R9_user_reported_result.json').read_text(encoding='utf-8'))
    old=json.loads((R9/'report.json').read_text(encoding='utf-8'))['provenance']['source_sha256']
    for name in DEPENDENCIES:
        if digest(R9/name)!=old[name]: raise RuntimeError('Frozen dependency changed: '+name)
    provenance.update(R9_C_sha256=BASE_SHA,
                      source_sha256={n:digest(HERE/n) for n in SOURCES},
                      R9_dependency_sha256={n:digest(R9/n) for n in DEPENDENCIES},
                      experiment_sha256=digest(HERE/'experiment_summary.json'),
                      feedback_sha256=digest(HERE/'R9_user_reported_result.json'))
    verifying=args.verify_only or args.reproduce_check
    with threadpool_limits(limits=1):
        if verifying:
            saved=json.loads((HERE/'report.json').read_text(encoding='utf-8'))
            if provenance!=saved['provenance'] or settings!=saved['settings']: raise RuntimeError('Provenance changed')
            score,validation=validate_package(b,cfg)
            if validation!=saved['validation'] or score['facets']!=saved['score']['facets']: raise RuntimeError('Release changed')
            if args.reproduce_check:
                generated,_=synthesize(b,cfg,**settings)
                if payload(generated)!=(HERE/'C.csv').read_bytes(): raise RuntimeError('Reproduction failed')
        else:
            if any((HERE/p).exists() for p in ['C.csv','report.json','submission/'+ZIP_NAME]): raise RuntimeError('Refusing to overwrite release')
            c,audit=synthesize(b,cfg,**settings)
            data=payload(c)
            if hashlib.sha256(data).hexdigest()!=chosen['csv_sha256']: raise RuntimeError('Selected candidate bytes did not reproduce')
            verify_fixed(load_base(),pd.read_csv(io.BytesIO(data),float_precision='round_trip'))
            score,_=evaluate_frame(pd.read_csv(io.BytesIO(data),float_precision='round_trip'),b,cfg)
            release_gate(score,feedback['utility_facets'])
            if score['facets']!=chosen['score']['facets']: raise RuntimeError('Selected score did not reproduce')
            (HERE/'C.csv').write_bytes(data); package(data)
            score,validation=validate_package(b,cfg)
            record=dict(model='R11',settings=settings,provenance=provenance,score=score,validation=validation,
                        audit=audit,utility_delta=score['utility']-min(feedback['utility_facets'].values()),
                        comparison_R9_official=feedback,selection_objective='maximize local utility; all four facets at least R9',
                        independent_holdout=False,formal_differential_privacy=False,official_attack_evaluated=False,
                        official_submission_performed=False,candidate_bytes_reproduced=True)
            (HERE/'report.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
        release_gate(score,feedback['utility_facets'])
        result=dict(verified=True,utility=score['utility'],facets=score['facets'],validation=validation,
                    candidate_bytes_reproduced=bool(not verifying or args.reproduce_check))
        target='verification.json' if not verifying else 'verification_recheck.json'
        (HERE/target).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
        print(json.dumps(result),flush=True)

if __name__=='__main__': main()
