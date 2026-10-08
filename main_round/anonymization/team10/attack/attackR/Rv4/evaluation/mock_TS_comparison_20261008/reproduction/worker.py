"""One public-input correction and fresh Rv3 fit; no private truth reads."""
import os,sys,json,hashlib,importlib.util
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(os.environ['RV4_REPOSITORY']);MOCK=ROOT/'main_round/mock_test'
sys.path.insert(0,str(MOCK))
import numpy as np
import pandas as pd
from mock_TS.common import public_tables,read_csv,write_json,sha256
from adapter import corrected
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main():
    wd=Path.cwd();tables=public_tables(wd)
    bm=read_csv(wd/'baseline/F_mia.csv');ba=read_csv(wd/'baseline/F_aia.csv')
    om=bm.set_index('record_id').loc[tables['candidates'].record_id,'0'].to_numpy()
    oa=ba.set_index('challenge_row_id').loc[tables['challenge'].challenge_row_id,'time_hat'].to_numpy()
    rv4=ROOT/'main_round/anonymization/team10/attack/attackR/Rv4'
    rv3=ROOT/'main_round/anonymization/team10/atack/Rv3'
    m4=load('_compare_rv4_numeric',rv4/'model.py')
    p3=load('_compare_rv3_plugin',rv3/'plugin.py')
    arguments={k:tables[k] for k in ['c','background','candidates','challenge','cfg']}
    guard={'active':True,'blocked':[]}
    def audit(event,args):
        if event!='open' or not guard['active'] or isinstance(args[0],int):return
        p=Path(os.fsdecode(args[0]))
        if p.suffix.lower() in ['.csv','.npz','.joblib','.pkl','.pickle','.parquet','.zip']:
            # Python package archives are runtime dependencies, not data inputs.
            absolute=p.resolve()
            if absolute.is_relative_to(ROOT/'main_round/attack/.venv') or absolute.is_relative_to(ROOT/'main_round/attack/.tools'):return
            guard['blocked'].append(str(absolute));raise RuntimeError('Data read during public-only inference')
    sys.addaudithook(audit)
    try:
        a4=corrected(m4,c=tables['c'],background=tables['background'],candidates=tables['candidates'],challenge=tables['challenge'],old_mia=om,old_aia=oa)
        a3=p3.predict(**arguments)
    finally:guard['active']=False
    if guard['blocked']:raise RuntimeError('Suppressed forbidden data read')
    for name,a in [('Rv3',a3),('Rv4',a4)]:
        out=wd/name;out.mkdir()
        pd.DataFrame({'record_id':tables['candidates'].record_id,'0':a['mia']}).to_csv(out/'F_mia.csv',index=False,float_format='%.17g')
        pd.DataFrame({'target_k':0,'challenge_row_id':tables['challenge'].challenge_row_id,'time_hat':a['aia']}).to_csv(out/'F_aia.csv',index=False,float_format='%.17g')
    write_json(wd/'worker.json',{'pid':os.getpid(),'fresh_process':True,'blocked_data_reads':guard['blocked'],'private_mask_applied':False,'Rv2_component':'verified cached same-case prediction','Rv3_fresh_fit':True,'Rv4_correction_recomputed':True,'input_files_sha256':{p.name:sha256(p) for p in wd.glob('*.csv')}})
if __name__=='__main__':main()
