"""Generate twice, compare and validate the local Rv4 submission."""
import argparse,subprocess,sys,os,io,zipfile,hashlib,hmac
import runtime as rt
import pandas as pd
from plugin import predict_world
from pathlib import Path
WORK=Path(os.environ.get('RV4_WORK_DIRECTORY',str(rt.MOCK/'local-output/Rv4-production'))).resolve()
def actual():
    common=rt.load('_rv4_actual_common',rt.ATTACK/'Attack_R/v1/common.py');a=common.load_actual()
    w,np,_,validator,cfg,dist,b,bg,cs,qs,*_=a
    read=lambda p:pd.read_csv(p,float_precision='round_trip')
    public=dict(candidates=b,self_labels=read(w.ORIGINALS/'B_self.csv'),background=bg,self_challenge=read(w.TARGETS/'aia_challenge_10.csv'),targets={k:dict(release=cs[k],challenge=qs[k]) for k in cs},schema=rt.read(w.ORIGINALS/'schema.json'),cfg=cfg)
    return a,public
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=str,default=str(rt.HERE/'submission'));p.add_argument('--pass-only',action='store_true');args=p.parse_args()
    from pathlib import Path
    out=Path(args.output).resolve()
    if out.exists():raise ValueError('Use a new output directory')
    if not out.is_relative_to(WORK if args.pass_only else rt.HERE):raise ValueError('Output escaped its allowed directory')
    if args.pass_only:
        a,public=actual();r=predict_world(**public);w,np,_,validator,cfg,dist,*_=a
        valid=w.validate_frames(np,pd,validator,r['F_mia'],r['F_aia'],cfg,dist)
        out.mkdir(parents=True)
        for n in ['F_mia','F_aia']:(out/(n+'.csv')).write_bytes(rt.csv(r[n]))
        rt.save(out/'verification.json',{'validation':valid,'diagnostics':r['diagnostics']});return
    if not rt.read(rt.HERE/'evaluation/confirmation/result.json')['completed']:raise ValueError('Confirmation required')
    from datetime import datetime,timezone
    work=WORK/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    for i in [1,2]:subprocess.run([sys.executable,'-B',str(rt.HERE/'run.py'),'--pass-only','--output',str(work/str(i))],check=True)
    content={n:(work/'1'/n).read_bytes() for n in ['F_mia.csv','F_aia.csv']}
    if any(x!=(work/'2'/n).read_bytes() for n,x in content.items()):raise ValueError('Independent runs differ')
    a,public=actual();w,np,_,validator,cfg,dist,*_=a
    frames={n:pd.read_csv(io.BytesIO(x),float_precision='round_trip') for n,x in content.items()}
    valid=w.validate_frames(np,pd,validator,frames['F_mia.csv'],frames['F_aia.csv'],cfg,dist)
    if any(rt.csv(frames[n])!=x for n,x in content.items()):raise ValueError('CSV roundtrip differs')
    out.mkdir(parents=True)
    for n,x in content.items():(out/n).write_bytes(x)
    zpath=out/'team10_Rv4_attack.zip'
    with zipfile.ZipFile(zpath,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n,x in content.items():z.writestr(n,x)
        z.writestr('token.txt',(w.ORIGINALS/'token.txt').read_bytes())
    with zipfile.ZipFile(zpath) as z:
        assert z.namelist()==['F_mia.csv','F_aia.csv','token.txt']
        assert all(z.read(n)==x for n,x in content.items())
        assert hmac.compare_digest(z.read('token.txt'),(w.ORIGINALS/'token.txt').read_bytes())
    check=validator.validate_submission_zip(zpath,cfg,w.TARGETS,'attack')
    if not check.ok:raise ValueError('; '.join(check.errors))
    changes={}
    for n,f in frames.items():
        old=pd.read_csv(rt.ATTACK/'Attack_R/v2/submission'/n,float_precision='round_trip')
        columns=[c for c in f if c not in ['record_id','target_k','challenge_row_id']]
        changes[n]=int((f[columns].to_numpy()!=old[columns].to_numpy()).sum())
    rt.save(out/'verification.json',{'completed':True,'two_independent_process_runs_byte_identical':True,'csv_roundtrip_byte_identical':True,'official_zip_validation':True,'actual_team10_validation':valid,'original_token_matches':True,'zip_members':['F_mia.csv','F_aia.csv','token.txt'],'sha256':{n:rt.sha(out/n) for n in [*content,zpath.name]},'changed_cells_from_Rv2':changes,'preservation':rt.preserve(),'official_submission_performed':False,'official_score_measured':False})
    print('Rv4 submission validated',zpath,flush=True)
if __name__=='__main__':main()
