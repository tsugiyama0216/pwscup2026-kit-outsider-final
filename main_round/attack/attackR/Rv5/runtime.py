import hashlib,importlib.util,json,sys,os
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
HERE=Path(__file__).resolve().parent;ATTACK=HERE.parents[1];ROOT=ATTACK.parents[1]
sys.path.insert(0,str(ATTACK))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    if Path(p).name.lower()=='token.txt':raise ValueError('No token hashing')
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def verify():
    m=read(HERE/'dependencies.json')
    for n,h in m['files'].items():
        if sha(ROOT/n)!=h:raise ValueError('Frozen file changed: '+n)
    return len(m['files'])
def actual():
    common=load('_rv5_common',ATTACK/'Attack_R/v1/common.py')
    return common.load_actual()
