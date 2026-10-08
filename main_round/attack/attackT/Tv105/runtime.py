"""Repository-anchored imports and immutable dependency verification."""
from pathlib import Path
import hashlib, importlib.util, json, sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ATTACK = ROOT / 'main_round/attack'
MOCK = ROOT / 'main_round/mock_test'
RV2 = ATTACK / 'Attack_R/v2'
RV4 = ROOT / 'main_round/anonymization/team10/attack/attackR/Rv4'
for p in (MOCK, ATTACK, ATTACK/'Attak_T'):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
import attack_research

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(s)
    sys.modules[name]=m; s.loader.exec_module(m); return m
def dependencies():
    original=json.loads((RV2/'production_dependencies.json').read_text(encoding='utf-8'))['files_sha256']
    for name,h in original.items():
        if sha(ROOT/name)!=h: raise ValueError('Frozen dependency changed: '+name)
    return original
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
