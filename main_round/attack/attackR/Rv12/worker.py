"""Public-only inference worker: verified same-world Rv2 or fresh frozen inference."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
MOCK=ROOT/'main_round/mock_test'
sys.path.insert(0,str(MOCK))
from rank_validation.adapters import _bootstrap, _load, sha, validate_public, predict_world
from rank_validation.model_worker import load_public, clean
from rank_validation.series_worker import load_baseline, verify_baseline_unchanged
from rank_validation.plugins import read_guard
from threadpoolctl import threadpool_limits


def save_json(path,value):
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def run(public_dir, output, baseline_dir=None, selected=None):
    public_dir,output=Path(public_dir).resolve(),Path(output).resolve()
    if output.exists():
        raise ValueError('Existing output is preserved')
    public,hashes=load_public(public_dir)
    _bootstrap()
    if baseline_dir is None:
        base=predict_world('Rv2',**public)
        aligned=base['F_mia'].set_index('record_id').loc[public['candidates'].record_id]
        bm={k:aligned[str(k)].to_numpy(float) for k in public['targets']}
        ba={k:base['F_aia'].loc[base['F_aia'].target_k.eq(k)].set_index('challenge_row_id').loc[item['challenge'].challenge_row_id].time_hat.to_numpy(float) for k,item in public['targets'].items()}
        binding={'baseline_computed_fresh':True,'frozen_baseline_diagnostics':base['diagnostics']}
    else:
        bm,ba,record=load_baseline(baseline_dir,public,hashes)
        binding={'baseline_output_sha256':record['output_sha256'],'baseline_diagnostics_sha256':record['diagnostics_sha256'],'baseline_files_unchanged':True,'baseline_model_signature':record['model_signature']}
    model=_load('_rv12_numeric',HERE/'model.py')
    mask_module=_load('_rv12_private_mask',ROOT/'main_round/attack/Attack_M/v1/actual.py')
    learned=_load('_rv12_learned',HERE/'learned_veto.py')
    learned_gate=learned.verify_dependencies()
    before={name:sha(HERE/name) for name in ('model.py','protocol.json','worker.py','learned_veto.py')}
    with read_guard(HERE/'model.py') as audit, threadpool_limits(limits=1):
        mask,proof=mask_module.derive_private_mask(public['candidates'].astype(str),public['self_labels'].astype(str),public['self_challenge'].astype(str),public['schema']['aia_qi_columns'],public['schema']['submitted_c_columns'])
        predictions=model.predict_all(candidates=public['candidates'],background=public['background'],targets=public['targets'],baseline_mia=bm,baseline_aia=ba,known_negative=mask)
        learned.allow_assets(audit)
        engine=learned.LearnedVeto()
        for name in learned.PENALTIES:
            predictions[name]={'F_mia':predictions['retain']['F_mia'].copy(), 'parts':[], 'diagnostics':{}}
        for k,item in public['targets'].items():
            choices=engine.predict(item['release'],public['background'],item['challenge'],ba[k])
            for name,(times,detail) in choices.items():
                predictions[name]['parts'].append(pd.DataFrame({'target_k':k,'challenge_row_id':item['challenge'].challenge_row_id.to_numpy(),'time_hat':times}))
                predictions[name]['diagnostics'][str(k)]=detail
        for name in learned.PENALTIES:
            predictions[name]['F_aia']=pd.concat(predictions[name].pop('parts'),ignore_index=True)
    if learned.verify_dependencies()!=learned_gate:
        raise ValueError('Learned dependencies changed')
    if audit['blocked'] or audit['code_changes']:
        raise RuntimeError('Unexpected inference reads')
    if any(not path.is_file() or sha(path)!=digest for path,digest in audit['code_hashes'].items()):
        raise ValueError('Imported source changed during inference')
    imported={path.relative_to(ROOT).as_posix():digest for path,digest in sorted(audit['code_hashes'].items())}
    if {name:sha(HERE/name) for name in before} != before or {name:sha(public_dir/name) for name in hashes} != hashes:
        raise ValueError('Inputs/source changed')
    if baseline_dir is not None:
        verify_baseline_unchanged(baseline_dir,record)
    if selected is not None:
        selection=json.loads(Path(selected).read_text(encoding='utf-8-sig'))
        if any(selection['source_sha256'].get(name) != digest for name,digest in before.items()):
            raise ValueError('Frozen numerical source differs')
        predictions={'Rv12':predictions[selection['aia']]}
    output.mkdir(parents=True,exist_ok=False)
    from sugiyama_v2.scoring import validate_predictions
    from rank_validation.replay import read_csv_strict
    for name,result in predictions.items():
        folder=output/name
        folder.mkdir()
        validate_predictions(result['F_mia'],result['F_aia'],{'team_id':10,**public})
        digests={}
        for key in ('F_mia','F_aia'):
            path=folder/(key+'.csv')
            path.write_bytes(result[key].to_csv(index=False,lineterminator='\n',float_format='%.17g').encode('utf8'))
            digests[path.name]=sha(path)
        validate_predictions(read_csv_strict(folder/'F_mia.csv'),read_csv_strict(folder/'F_aia.csv'),{'team_id':10,**public})
        save_json(folder/'diagnostics.json',{'model_id':name,'public_input_sha256':hashes,'output_sha256':digests,'input_files_unchanged':True,'source_sha256':before,'imported_source_sha256':imported,'imported_sources_unchanged':True,**binding,'private_proof':proof,'targets':result['diagnostics'],'public_only':True,'learned_dependency_verification':learned_gate,'blocked_reads':audit['blocked'],'official_submission_performed':False})
        print(json.dumps({'variant':name,'suppressed':sum(x['suppressed'] for x in result['diagnostics'].values()),'output':str(folder)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--public',required=True,type=Path)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--selection',type=Path)
    args=parser.parse_args()
    run(args.public,args.output,args.baseline,args.selection)


