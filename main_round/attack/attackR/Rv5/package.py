"""Two process reproductions and one local credential-bearing submission ZIP."""
import io,os,subprocess,sys,zipfile,hmac
import runtime as rt
import numpy as np
import pandas as pd

def main():
    out=rt.HERE/'submission'
    if out.exists():raise ValueError('Preserve prior submission')
    work=rt.HERE/'.work'
    rt.verify();env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
    for i in (1,2):
        subprocess.run([sys.executable,'-B',str(rt.HERE/'infer.py'),'--output',str(work/f'pass{i}')],check=True,env=env)
    data={n:(work/'pass1'/n).read_bytes() for n in ['F_mia.csv','F_aia.csv']}
    for n,b in data.items():
        if b!=(work/'pass2'/n).read_bytes():raise ValueError('Independent regeneration differs: '+n)
    actual=rt.actual();w,_,_,validator,cfg,dist,*_=actual
    frames={n:pd.read_csv(io.BytesIO(b),float_precision='round_trip') for n,b in data.items()}
    valid=w.validate_frames(np,pd,validator,frames['F_mia.csv'],frames['F_aia.csv'],cfg,dist)
    negative_checks={}
    for label in ['missing_mia','duplicate_mia','nan_mia','out_of_range','missing_aia','duplicate_aia','negative_time','bad_target']:
        m=frames['F_mia.csv'].copy();a=frames['F_aia.csv'].copy()
        if label=='missing_mia':m=m.iloc[:-1]
        elif label=='duplicate_mia':m.loc[1,'record_id']=m.loc[0,'record_id']
        elif label=='nan_mia':m.iloc[0,1]=np.nan
        elif label=='out_of_range':m.iloc[0,1]=1.1
        elif label=='missing_aia':a=a.iloc[:-1]
        elif label=='duplicate_aia':a.loc[1,'challenge_row_id']=a.loc[0,'challenge_row_id']
        elif label=='negative_time':a.loc[0,'time_hat']=-1
        else:a.loc[0,'target_k']=999
        try:w.validate_frames(np,pd,validator,m,a,cfg,dist)
        except (ValueError,AssertionError):negative_checks[label]=True
        else:raise ValueError('Bad input not rejected: '+label)
    for n,b in data.items():
        if frames[n].to_csv(index=False,lineterminator='\n',float_format='%.17g').encode()!=b:raise ValueError('CSV roundtrip differs')
    rt.verify()
    for n,h in rt.read(rt.HERE/'preservation.json')['files'].items():
        if rt.sha(rt.ROOT/n)!=h:raise ValueError('Prior R artifact changed: '+n)
    out.mkdir()
    for n,b in data.items():(out/n).write_bytes(b)
    z=out/'team10_Rv5_attack.zip'
    with zipfile.ZipFile(z,'x',compression=zipfile.ZIP_DEFLATED) as arc:
        for n,b in data.items():arc.writestr(n,b)
        arc.writestr('token.txt',(w.ORIGINALS/'token.txt').read_bytes())
    with zipfile.ZipFile(z) as arc:
        assert arc.namelist()==['F_mia.csv','F_aia.csv','token.txt'] and arc.testzip() is None
        assert all(arc.read(n)==b for n,b in data.items())
        assert hmac.compare_digest(arc.read('token.txt'),(w.ORIGINALS/'token.txt').read_bytes())
    check=validator.validate_submission_zip(z,cfg,w.TARGETS,'attack')
    if not check.ok:raise ValueError('Official package validation failed')
    report=rt.read(work/'pass1/report.json')
    rt.write(rt.HERE/'production_diagnostics.json',report)
    rt.write(rt.HERE/'verification.json',{'model':'Rv5','completed':True,'validation':valid,'two_independent_runs_byte_identical':True,
        'csv_roundtrip_identical':True,'official_zip_valid':True,'token_bytes_match_original':True,'negative_checks':negative_checks,
        'prior_R_preservation_files':len(rt.read(rt.HERE/'preservation.json')['files']),
        'prediction_sha256':{n:rt.sha(out/n) for n in data},'zip_sha256':rt.sha(z),'official_score_measured':False,
        'mock_scores_used':False,'official_submission_performed':False})
    print('Validated local Rv5 ZIP:',z,flush=True)
if __name__=='__main__':main()
