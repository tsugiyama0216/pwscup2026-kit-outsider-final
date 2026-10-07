"""Read-only review of the scoped Git index, never printing credentials."""
from pathlib import Path
import subprocess,json,hashlib,argparse
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
PREFIX='main_round/anonymization/team10/atack/Rv3/'
def main():
    p=argparse.ArgumentParser();p.add_argument('--tree',type=Path,required=True);a=p.parse_args()
    def git(*args):return subprocess.check_output(['git','-C',str(a.tree),*args])
    names=git('diff','--cached','--name-only','-z').decode().split('\0');names=[n for n in names if n]
    if not names or any(not n.startswith(PREFIX) for n in names):raise ValueError('Unrelated staged changes')
    token=(ROOT/'main_round/anonymization/team10/participant_data/token.txt').read_bytes().strip()
    if not token:raise ValueError('Empty credential')
    for name in names:
        if name.endswith('.zip') or Path(name).name=='token.txt':raise ValueError('Credential-bearing artifact staged')
        data=git('show',':'+name)
        if token in data:raise ValueError('Credential content detected in staged artifact')
        if data!=(ROOT/name).read_bytes():raise ValueError('Staged bytes differ from local artifact: '+name)
    deps=json.loads((HERE/'production_dependencies.json').read_text())['files_sha256']
    for name,expected in deps.items():
        actual=hashlib.sha256(git('show',':'+name)).hexdigest()
        if actual!=expected:raise ValueError('Published dependency differs: '+name)
    result={'staged_files_before_this_record':len(names),'scope_only_Rv3':True,'live_credential_scan_passed':True,'submission_ZIPs_staged':0,'local_staged_bytes_equal':True,'production_dependencies_verified_in_index':len(deps),'official_submission_performed':False}
    print(json.dumps(result))
    (HERE/'publication_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
