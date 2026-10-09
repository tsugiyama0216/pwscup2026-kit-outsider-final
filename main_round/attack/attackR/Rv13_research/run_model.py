"""Reproduce one frozen independent model on the official public main inputs."""
from pathlib import Path
import sys,os,argparse,json,hashlib,shutil,subprocess
ROOT=Path(__file__).resolve().parents[4];MOCK=ROOT/'main_round/mock_test';sys.path.insert(0,str(MOCK))
p=argparse.ArgumentParser();p.add_argument('--model',choices=['Rv13','Rv14','Rv15'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
out=a.output.resolve();allowed=(MOCK/'local-output').resolve()
if not out.is_relative_to(allowed) or out==allowed or out.exists():p.error('Use a new directory below main_round/mock_test/local-output')
model=ROOT/'main_round/attack/attackR'/a.model;selected=json.loads((model/'selection.json').read_text())
if hashlib.sha256((model/'model.py').read_bytes()).hexdigest()!=selected['source_sha256']:raise ValueError('Frozen model changed')
from rank_validation.panel import workspace
originals=workspace.verify_hashes();out.mkdir(parents=True);pub=out/'public';pub.mkdir()
for name in ['B.csv','B_self.csv','A_bg.csv','schema.json']:shutil.copyfile(workspace.ORIGINALS/name,pub/name)
shutil.copyfile(workspace.TARGETS/'aia_challenge_10.csv',pub/'aia_challenge_10.csv');(pub/'targets').mkdir()
for k in list(range(10))+list(range(11,18))+[22,25,26,27,28]:
 for name in [f'C_{k}.csv',f'aia_challenge_{k}.csv']:shutil.copyfile(workspace.TARGETS/name,pub/'targets'/name)
_,_,_,cfg,_=workspace.load_runtime();(pub/'cfg.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8')
env=os.environ.copy();env.update(PYTHONUTF8='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
cmd=[sys.executable,'-B','-m','rank_validation.model_worker','--plugin',str(model/'model.py'),'--name',a.model,'--public',str(pub),'--output',str(out/'predictions')]
with (out/'run.log').open('x',encoding='utf8') as log:subprocess.run(cmd,cwd=MOCK,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
if workspace.verify_hashes()!=originals:raise ValueError('Original inputs changed')
print('Verified local predictions:',out/'predictions')
