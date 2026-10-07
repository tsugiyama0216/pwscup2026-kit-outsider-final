"""Generate twice, validate real IDs, and create the local submission ZIP."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import argparse,json,sys,subprocess,hashlib,zipfile,hmac,io,importlib.util
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4];ATTACK=ROOT/'main_round/attack'
sys.path.insert(0,str(ATTACK))
sys.path.insert(0,str(ATTACK/'official-kit/starter/src'))
import numpy as np
import pandas as pd
import attack_workspace as w
from selected import predict_selected

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    if p.name=='token.txt':raise ValueError('Never hash token')
    return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def csv(f):return f.to_csv(index=False,lineterminator='\n',float_format='%.17g').encode()
def frozen():
    s=read(HERE/'selection.json')
    for n,h in s['source_sha256'].items():
        if sha(HERE/n)!=h:raise ValueError('Selected source mismatch: '+n)
    for n,h in read(HERE/'production_sources.json').items():
        if sha(HERE/n)!=h:raise ValueError('Frozen production source mismatch: '+n)
    for n,h in read(HERE/'production_dependencies.json')['files_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Frozen dependency mismatch: '+n)
    if not read(HERE/'confirmation.json')['completed']:raise ValueError('Confirmation required')
    return s
def runtime():
    w.verify_hashes()
    _,_,modules,cfg,versions=w.load_runtime()
    validator=modules['pwscup2026.kit.validate']
    *_,dist=w.load_inputs(np,pd,validator)
    return validator,cfg,versions,dist
def infer(output):
    selection=frozen();validator,cfg,versions,dist=runtime()
    get=lambda p:pd.read_csv(p,float_precision='round_trip')
    b=get(w.ORIGINALS/'B.csv');bg=get(w.ORIGINALS/'A_bg.csv')
    spec=importlib.util.spec_from_file_location('_rv3_public_mask',ATTACK/'Attack_M/v1/actual.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    schema=read(w.ORIGINALS/'schema.json')
    strings=lambda p:pd.read_csv(p,dtype=str)
    mask,proof=module.derive_private_mask(strings(w.ORIGINALS/'B.csv'),strings(w.ORIGINALS/'B_self.csv'),strings(w.TARGETS/'aia_challenge_10.csv'),schema['aia_qi_columns'],schema['submitted_c_columns'])
    fm=pd.DataFrame({'record_id':b.record_id});parts=[]
    for k in w.ATTACK_TARGETS:
        q=get(w.TARGETS/f'aia_challenge_{k}.csv')
        a=predict_selected(selection,c=get(w.TARGETS/f'C_{k}.csv'),background=bg,candidates=b,challenge=q,cfg=cfg)
        scores=a['mia'].copy();scores[mask]=0;fm[str(k)]=scores
        parts.append(pd.DataFrame({'target_k':k,'challenge_row_id':q.challenge_row_id,'time_hat':a['aia']}))
        print('Rv3 target',k,'complete',flush=True)
    fa=pd.concat(parts,ignore_index=True)
    valid=w.validate_frames(np,pd,validator,fm,fa,cfg,dist)
    output.mkdir(parents=True,exist_ok=False)
    for n,f in [('F_mia.csv',fm),('F_aia.csv',fa)]: (output/n).write_bytes(csv(f))
    save(output/'validation.json',{'validation':valid,'private_mask':proof,'versions':versions})

def main():
    p=argparse.ArgumentParser();p.add_argument('--infer',type=Path);p.add_argument('--output',type=Path,default=HERE/'submission');args=p.parse_args()
    if args.infer:infer(args.infer);return
    output=args.output.resolve()
    if not output.is_relative_to(HERE) or output.exists():raise ValueError('Use a new output under Rv3')
    frozen()
    work=ROOT/'main_round/mock_test/local-output/Rv3-production'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    for i in [1,2]:subprocess.run([sys.executable,'-B',str(HERE/'run.py'),'--infer',str(work/str(i))],check=True)
    contents={n:(work/'1'/n).read_bytes() for n in ['F_mia.csv','F_aia.csv']}
    for n,v in contents.items():
        if v!=(work/'2'/n).read_bytes():raise ValueError('Independent runs differ')
        if csv(pd.read_csv(io.BytesIO(v),float_precision='round_trip'))!=v:raise ValueError('CSV roundtrip differs')
    validator,cfg,versions,dist=runtime()
    output.mkdir(parents=True)
    for n,v in contents.items():(output/n).write_bytes(v)
    zipped=output/'team10_Rv3_attack.zip'
    with zipfile.ZipFile(zipped,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n,v in contents.items():z.writestr(n,v)
        z.writestr('token.txt',(w.ORIGINALS/'token.txt').read_bytes())
    with zipfile.ZipFile(zipped) as z:
        assert z.namelist()==['F_mia.csv','F_aia.csv','token.txt']
        assert hmac.compare_digest(z.read('token.txt'),(w.ORIGINALS/'token.txt').read_bytes())
        for n,v in contents.items():assert z.read(n)==v
    check=validator.validate_submission_zip(zipped,cfg,w.TARGETS,'attack')
    if not check.ok:raise ValueError('Official ZIP validation failed: '+'; '.join(check.errors))
    pkg=validator.check_package(zipped,cfg);pkg.token=None
    valid=w.validate_frames(np,pd,validator,pkg.frames['F_mia.csv'],pkg.frames['F_aia.csv'],cfg,dist)
    frozen()
    receipt=dict(model='Rv3',completed=True,utc=datetime.now(timezone.utc).isoformat(),official_zip_and_real_id_validation=valid,
        two_independent_runs_byte_identical=True,csv_roundtrip_byte_identical=True,original_token_bytes_match=True,
        file_sha256={n:sha(output/n) for n in [*contents,zipped.name]},inference=read(work/'1/validation.json'),
        official_score_measured=False,official_submission_performed=False,official_attempts_consumed=0,zip_uploaded=False)
    save(output/'verification.json',receipt)
    print('Validated:',zipped,flush=True)
if __name__=='__main__':main()
