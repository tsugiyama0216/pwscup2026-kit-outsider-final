"""Verify Rv12 reproduction, official CSV layout, retained inputs and Rv8 parity."""
import argparse,json,subprocess,hashlib
from pathlib import Path
from datetime import datetime,timezone
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'main_round/mock_test'))
from rank_validation.adapters import _bootstrap,sha
from rank_validation.replay import read_csv_strict
from rank_validation.verify_adapter_parity import compare_csv
_bootstrap()
import attack_workspace as workspace


def verify(first,second,output):
    import numpy as np
    from scipy.stats import rankdata
    first,second,output=map(lambda p:Path(p).resolve(),(first,second,output))
    if output.exists():raise ValueError('Preserve validation record')
    selection=json.loads((HERE/'selection.json').read_text(encoding='utf8'))
    for name,digest in selection['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Fixed source mismatch: '+name)
    original=workspace.verify_hashes()
    np,pd,modules,cfg,versions=workspace.load_runtime()
    validator=modules['pwscup2026.kit.validate']
    inputs=workspace.load_inputs(np,pd,validator)
    reports=[]
    for run in (first,second):
        folder=run/'predictions/Rv12'
        m,a=[read_csv_strict(folder/n) for n in ('F_mia.csv','F_aia.csv')]
        validation=workspace.validate_frames(np,pd,validator,m,a,cfg,inputs[4])
        diag=json.loads((folder/'diagnostics.json').read_text(encoding='utf8'))
        if not diag['public_only'] or diag['blocked_reads'] or not diag['imported_sources_unchanged']:
            raise ValueError('Invalid inference provenance')
        if diag['private_proof']['exact_private_rows']!=163:raise ValueError('Private-mask proof differs')
        for path,digest in diag['imported_source_sha256'].items():
            if sha(ROOT/path)!=digest:raise ValueError('Imported source changed after inference')
        if (m.drop(columns='record_id').eq(0).all(axis=1)).sum()!=163:raise ValueError('Known-negative mask differs')
        reports.append({'run':str(run),'validation':validation,'all_163_known_negatives_zero':True,'production_verification_sha256':sha(run/'verification.json'),'diagnostics_sha256':sha(folder/'diagnostics.json')})
    byte_parity={n:compare_csv(first/'predictions/Rv12'/n,second/'predictions/Rv12'/n) for n in ('F_mia.csv','F_aia.csv')}
    if not all(v['byte_identical'] for v in byte_parity.values()):raise ValueError('Reproduction bytes differ')
    folder=first/'predictions/Rv12'
    rv8=ROOT/'main_round/anonymization/team10/attack/attackR/Rv8/submission'
    inherited={n:compare_csv(folder/n,rv8/n) for n in ('F_mia.csv','F_aia.csv')}
    current=read_csv_strict(folder/'F_mia.csv').set_index('record_id')
    old=read_csv_strict(rv8/'F_mia.csv').set_index('record_id').loc[current.index,current.columns]
    rank_parity={str(k):bool(np.array_equal(rankdata(current[k]),rankdata(old[k]))) for k in current.columns}
    if not all(rank_parity.values()):raise ValueError('Expected Rv8 rank equivalence was not reproduced')
    rv2=ROOT/'main_round/attack/Attack_R/v2/submission'
    aia_parity=compare_csv(folder/'F_aia.csv',rv2/'F_aia.csv')
    if not aia_parity['byte_identical']:raise ValueError('Retained AIA differs')
    old2=read_csv_strict(rv2/'F_mia.csv').set_index('record_id').loc[current.index,current.columns]
    rank_changed={str(k):int(np.count_nonzero(rankdata(current[k])!=rankdata(old2[k]))) for k in current.columns}
    diff=subprocess.run(['git','diff','--exit-code'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if diff.returncode:raise ValueError('Existing tracked file changed')
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'selection_sha256':sha(HERE/'selection.json'),'fixed_sources_unchanged':True,'official_originals':original,'official_csv_validation':reports,'two_fresh_runs_byte_identical':True,'reproduction':byte_parity,'Rv8_reference':inherited,'Rv8_rank_parity':rank_parity,'Rv2_aia_parity':aia_parity,'Rv2_rank_positions_changed':rank_changed,'existing_tracked_files_unchanged':True,'dependency_versions':versions,'official_submission_performed':False,'github_push_performed':False,'credential_package_created':False,'verifier_sha256':sha(__file__)}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'output':str(output),'two_runs_byte_identical':True,'Rv8_rank_parity':all(rank_parity.values()),'MIA_rank_positions_changed':sum(rank_changed.values()),'AIA_unchanged':True,'official_validation':True}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--first',required=True,type=Path);p.add_argument('--second',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();verify(a.first,a.second,a.output)
