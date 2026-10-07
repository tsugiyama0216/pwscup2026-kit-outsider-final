"""Record and verify noncredential source dependencies and frozen prior versions."""
from pathlib import Path
import hashlib,json,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
def sha(p):
    if p.name=='token.txt':raise ValueError('No token hashes')
    return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    prior=ROOT/'main_round/attack/Attack_R/v2'
    hashes=json.loads((prior/'production_dependencies.json').read_text())['files_sha256']
    for name,value in hashes.items():
        if sha(ROOT/name)!=value:raise ValueError('Frozen dependency changed: '+name)
    for version in ['v1','v2f','v2']:
        folder=prior.parent/version
        record=json.loads((folder/'verification.json').read_text(encoding='utf-8'))
        for name,value in record['file_sha256'].items():
            p=folder/'submission'/name
            if p.suffix=='.zip' and not p.exists():continue
            if sha(p)!=value:raise ValueError('Frozen artifact changed: '+str(p))
    result={'frozen_Rv2_dependency_files_verified':len(hashes),'frozen_Rv1_Rv2f_Rv2_submissions_unchanged':True}
    print(json.dumps(result));return result
if __name__=='__main__':
    result=verify()
    if len(sys.argv)>1:
        out=HERE/sys.argv[1]
        if out.exists():raise ValueError('Preserve audit record')
        out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
