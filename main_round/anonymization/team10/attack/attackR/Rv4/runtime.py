"""Resolve repository and preserve the frozen Rv2 dependency closure."""
import os, sys, json, hashlib, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
def root():
    if os.environ.get('RV4_REPOSITORY'):
        return Path(os.environ['RV4_REPOSITORY']).resolve()
    for p in HERE.parents:
        if (p/'main_round/attack/Attack_R/v2/model.py').exists(): return p
    raise RuntimeError('Set RV4_REPOSITORY to the existing repository')
ROOT = root()
ATTACK = ROOT/'main_round/attack'
MOCK = ROOT/'main_round/mock_test'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
for p in (ATTACK, ATTACK/'Attak_T', MOCK): sys.path.insert(0,str(p))
import attack_research
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def sha(p):
    if Path(p).name.lower()=='token.txt': raise ValueError('Do not hash credentials')
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
    from sugiyama_v2.io import save_json
    Path(p).parent.mkdir(parents=True,exist_ok=True);save_json(p,x)
def csv(frame): return frame.to_csv(index=False,lineterminator='\n',float_format='%.17g').encode()
def preserve():
    manifest=read(ATTACK/'Attack_R/v2/production_dependencies.json')['files_sha256']
    for name,h in manifest.items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen dependency changed: '+name)
    return {'verified_files':len(manifest),'manifest_sha256':sha(ATTACK/'Attack_R/v2/production_dependencies.json')}
