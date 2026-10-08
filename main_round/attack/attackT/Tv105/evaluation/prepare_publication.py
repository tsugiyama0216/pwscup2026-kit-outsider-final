"""Copy only explicitly scoped non-credential pl2 artifacts to its worktree."""
from pathlib import Path
import shutil,json,hashlib,subprocess,sys
ROOT=Path(__file__).resolve().parents[5]
DEST=Path('C:/Users/Owner/.codex/worktrees/pl2-tv101-104/pws2026abe2')
SCOPES=[Path('main_round/attack/attackT')/n for n in ['pl2','Tv101','Tv102','Tv103','Tv104']]
ADAPTER=Path('main_round/attack/Attack_R/v2/mock_TS_evaluation_20261007/adapter.py')

def main():
    if not (DEST/'.git').is_file():raise ValueError('Expected task worktree')
    sources=[]
    for rel in SCOPES:
        for p in (ROOT/rel).rglob('*'):
            if not p.is_file() or p.name=='publication_files.json':continue
            if any(x.startswith('reproduced') or x=='__pycache__' for x in p.parts):continue
            if p.suffix.lower() in ('.zip','.pyc') or p.name.lower()=='token.txt':continue
            if p.suffix.lower() not in ('.py','.json','.csv','.html','.md','.ps1') and p.name!='.gitignore':raise ValueError('Unexpected artifact '+str(p))
            sources.append(p)
    sources.append(ROOT/ADAPTER)
    token=(ROOT/'main_round/anonymization/team10/participant_data/token.txt').read_bytes().strip()
    manifest={}
    for source in sources:
        data=source.read_bytes()
        if token and token in data:raise ValueError('Credential detected in publication scope')
        rel=source.relative_to(ROOT);target=DEST/rel
        if not target.resolve().is_relative_to(DEST.resolve()):raise ValueError('Scope escaped destination')
        tracked=subprocess.run(['git','-C',str(DEST),'ls-files','--error-unmatch',str(rel)],capture_output=True).returncode==0
        if tracked and target.read_bytes()!=data:raise ValueError('Would overwrite preexisting tracked file '+str(rel))
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        manifest[str(rel).replace('\\','/')]=hashlib.sha256(data).hexdigest()
    dest=ROOT/'main_round/attack/attackT/pl2/evaluation/publication_files.json'
    dest.write_text(json.dumps({'files':manifest,'live_token_scan_passed':True,'token_files_or_zips_included':False},indent=2)+'\n',encoding='utf-8')
    target=DEST/dest.relative_to(ROOT);shutil.copyfile(dest,target)
    print('Scoped non-credential files copied:',len(manifest)+1)
if __name__=='__main__':main()
