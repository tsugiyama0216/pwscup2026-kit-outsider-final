"""Create Rv12 CSVs from actual public inputs; never reads/copies a credential."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
MOCK=ROOT/'main_round/mock_test'
sys.path.insert(0,str(MOCK))
from rank_validation.adapters import _bootstrap,sha,TARGETS
_bootstrap()
import attack_workspace as workspace
from rank_validation.model_worker import load_public
from rank_validation.verify_adapter_parity import compare_csv


def generate(output):
    output=Path(output).resolve()
    allowed=(MOCK/'local-output').resolve()
    if output==allowed or not output.is_relative_to(allowed):
        raise ValueError('Use a new output directory under main_round/mock_test/local-output')
    if output.exists():
        raise ValueError('Existing output is preserved; choose a new directory')
    selection=json.loads((HERE/'selection.json').read_text(encoding='utf-8-sig'))
    for name,digest in selection['source_sha256'].items():
        if sha(HERE/name)!=digest:
            raise ValueError('Frozen Rv12 source changed: '+name)
    verification=workspace.verify_hashes()
    _,_,_,cfg,_=workspace.load_runtime()
    output.mkdir(parents=True)
    public=output/'public'
    (public/'targets').mkdir(parents=True)
    participant=ROOT/'main_round/anonymization/team10/participant_data'
    for name in ('B.csv','B_self.csv','A_bg.csv','schema.json'):
        shutil.copyfile(participant/name,public/name)
    shutil.copyfile(ROOT/'main_round/attack/targets/aia_challenge_10.csv',public/'aia_challenge_10.csv')
    for k in TARGETS:
        for name in (f'C_{k}.csv',f'aia_challenge_{k}.csv'):
            shutil.copyfile(ROOT/'main_round/attack/targets'/name,public/'targets'/name)
    (public/'cfg.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf8')
    _,hashes=load_public(public)
    baseline=output/'Rv2'
    command=[sys.executable,'-B','-m','rank_validation.model_worker','--model','Rv2','--public',str(public),'--output',str(baseline)]
    subprocess.run(command,cwd=MOCK,check=True)
    reference=ROOT/'main_round/attack/Attack_R/v2/submission'
    parity={name:compare_csv(baseline/name,reference/name) for name in ('F_mia.csv','F_aia.csv')}
    if not all(x['numeric_identical'] for x in parity.values()):
        raise ValueError('Fresh Rv2 differs from frozen production baseline')
    subprocess.run([sys.executable,'-B',str(HERE/'worker.py'),'--public',str(public),'--baseline',str(baseline),'--output',str(output/'predictions'),'--selection',str(HERE/'selection.json')],cwd=ROOT,check=True)
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'model':'Rv12','selection_sha256':sha(HERE/'selection.json'),'public_input_sha256':hashes,'original_verification':verification,'Rv2_parity':parity,'csv_sha256':{name:sha(output/'predictions/Rv12'/name) for name in ('F_mia.csv','F_aia.csv')},'official_submission_performed':False,'credentials_read':False}
    (output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'output':str(output/'predictions/Rv12'),'csv_sha256':report['csv_sha256']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    generate(args.output)

