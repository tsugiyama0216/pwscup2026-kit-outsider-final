"""Fresh process for one anonymous public bundle; no truth in prediction."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,json,sys,time
from pathlib import Path
import runtime as rt
from mock_TS.common import public_tables
import pandas as pd
from model import predict_all

def main():
    t=time.monotonic();tables=public_tables(Path.cwd())
    schema=json.loads(Path('schema.json').read_text(encoding='utf-8-sig'))
    deps=rt.dependencies()
    adapter=rt.load('_pl2_guard',rt.RV2/'mock_TS_evaluation_20261007/adapter.py')
    assets=[rt.ROOT/n for n in deps if n.startswith('main_round/attack/Attak_T/local-output/')
            and Path(n).suffix in ('.json','.joblib')]
    with adapter._input_guard(assets) as audit:
        answers,diag=predict_all(**{k:tables[k] for k in ('c','background','candidates','challenge','cfg')},schema=schema)
    if audit['blocked']:raise RuntimeError('Forbidden data read')
    for name,a in answers.items():
        dest=Path('predictions')/name;dest.mkdir(parents=True,exist_ok=False)
        pd.DataFrame({'record_id':tables['candidates'].record_id,'0':a['mia']}).to_csv(dest/'F_mia.csv',index=False,float_format='%.17g')
        pd.DataFrame({'target_k':0,'challenge_row_id':tables['challenge'].challenge_row_id,'time_hat':a['aia']}).to_csv(dest/'F_aia.csv',index=False,float_format='%.17g')
    diag.update(pid=os.getpid(),seconds=time.monotonic()-t,blocked_data_reads=audit['blocked'],asset_reads=sorted(audit['asset_reads']))
    rt.write(Path('predictions')/'diagnostics.json',diag)
if __name__=='__main__':main()
