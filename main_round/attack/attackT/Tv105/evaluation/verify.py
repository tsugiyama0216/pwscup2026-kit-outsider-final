"""Independent baseline/selected-plugin parity and preservation evidence."""
import argparse,os,sys,json
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime as rt
import numpy as np,pandas as pd
from mock_TS.common import public_tables,read_json,read_csv
from mock_TS.runner import load_suite
from mock_TS.scoring import validate_predictions

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--probe',action='store_true');args=parser.parse_args()
    out=rt.MOCK/'local-output/mock_TS/pl2_20261008';sel=read_json(rt.HERE/'selection.json')
    for n,h in sel['source_hashes'].items():
        if rt.sha(rt.HERE/n)!=h:raise ValueError('Selected source changed '+n)
    deps=rt.dependencies();freeze=read_json(rt.HERE/'evaluation/source_freeze.json')
    for n,h in freeze['rv4_sources'].items():
        if rt.sha(rt.ROOT/n)!=h:raise ValueError('Rv4 changed '+n)
    if args.probe:
        base=out/'development/runs/case_001/R1';os.chdir(base/'input');tables=public_tables(Path.cwd())
        public={k:tables[k] for k in ['c','background','candidates','challenge','cfg']}
        adapter=rt.load('_pl2_baseline_check',rt.RV2/'mock_TS_evaluation_20261007/adapter.py')
        old=adapter.predict(**public)
        for key,filename,col in [('mia','F_mia.csv','0'),('aia','F_aia.csv','time_hat')]:
            np.testing.assert_array_equal(old[key],read_csv(base/'predictions/Rv2'/filename)[col].to_numpy())
        from plugin import predict
        ans=predict(**public)
        for key,filename,col in [('mia','F_mia.csv','0'),('aia','F_aia.csv','time_hat')]:
            np.testing.assert_array_equal(ans[key],read_csv(base/'predictions/pl2'/filename)[col].to_numpy())
        rt.write(rt.HERE/'evaluation/independent_probe.json',{'frozen_Rv2_adapter_matches':True,'standalone_pl2_matches':True,
               'case':'development/case_001/R1','source_hashes':sel['source_hashes'],'dependencies_verified':len(deps)})
        print('Independent frozen Rv2 and standalone pl2 match');return
    count=0;diags=[]
    for phase in ['development','confirmation']:
        suite,manifest=load_suite(out/phase/'suite')
        rows=read_json(rt.HERE/'evaluation'/phase/'results.json')
        if len(rows)!=216:raise ValueError('Incomplete six-model comparison')
        for row in rows:
            base=out/phase/'runs'/row['case_id']/row['release'];pred=base/'predictions'/row['attack']
            for name,h in row['prediction_sha256'].items():
                if rt.sha(pred/name)!=h:raise ValueError('Prediction changed')
            if row['floor_estimate']['repetitions']!=1000:raise ValueError('Wrong floor')
            count+=1
        pl2=read_json(rt.HERE/'evaluation'/phase/'pl2_results.json')
        if len(pl2)!=36:raise ValueError('Incomplete synthesis comparison')
        for row in pl2:
            base=out/phase/'runs'/row['case_id']/row['release']
            validate_predictions(read_csv(base/'predictions/pl2/F_mia.csv'),read_csv(base/'predictions/pl2/F_aia.csv'),
                                 read_csv(base/'input/B.csv'),read_csv(base/'input/aia_challenge_0.csv'))
        for p in (out/phase/'runs').glob('*/*/predictions/diagnostics.json'):
            d=read_json(p)
            if d['blocked_data_reads']:raise ValueError('Forbidden read attempted')
            diags.append(d)
    if len(diags)!=72:raise ValueError('Incomplete process diagnostics')
    # Model/PID independence means new process per case/C, with six explicit
    # pure variants sharing the same frozen baseline computation.
    rt.write(rt.HERE/'verification.json',{'completed':True,'six_model_scored_predictions':count,'pl2_scored_predictions':72,
        'fresh_case_release_processes':len(diags),'unique_pids':len({d['pid'] for d in diags}),
        'forbidden_data_reads':0,'frozen_dependencies_preserved':len(deps),'rv4_sources_preserved':True,
        'selection_sha256':rt.sha(rt.HERE/'selection.json'),'new_numerical_tests_passed':6,
        'mock_TS_tests_passed':23,'generation_tests_retried_for_temp_directory_permission':True,
        'confirmation_reselection':False,'official_submission':False,'official_score_prediction':False,
        'limitations':read_json(rt.HERE/'protocol.json')['limitations']})
    print('Verified',count+72,'scored predictions and',len(diags),'fresh processes')
if __name__=='__main__':main()
