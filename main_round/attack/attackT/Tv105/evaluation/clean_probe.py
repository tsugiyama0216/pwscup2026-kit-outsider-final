"""Run selected plugin from a clean checkout against a public bundle."""
from pathlib import Path
import argparse,sys,os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime as rt
import numpy as np
from mock_TS.common import public_tables,read_csv
from plugin import predict
import attack_research,local_eval

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    for m in [rt,attack_research,local_eval]:
        if not Path(m.__file__).resolve().is_relative_to(rt.ROOT):raise ValueError('Import escaped clean checkout')
    tables=public_tables(a.bundle);args={k:tables[k] for k in ['c','background','candidates','challenge','cfg']}
    result=predict(**args)
    for key,name,col in [('mia','F_mia.csv','0'),('aia','F_aia.csv','time_hat')]:
        np.testing.assert_array_equal(result[key],read_csv(a.predictions/name)[col].to_numpy())
    rt.write(a.receipt,{'clean_checkout_inference_matches':True,'root':str(rt.ROOT),'source_imports_verified':True,
                      'dependencies_verified':len(rt.dependencies()),'another_physical_PC_tested':False})
    print('Clean checkout pl2 parity passed')
if __name__=='__main__':main()
