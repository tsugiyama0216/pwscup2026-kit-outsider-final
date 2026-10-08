"""Generate reproducible main-round pl1 predictions and a local submission ZIP."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from model import HERE, ATTACK, predict_world
import pandas as pd
import attack_workspace as w


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def actual():
    w.verify_hashes()
    np, pd, modules, cfg, versions = w.load_runtime()
    *_, dist = w.load_inputs(np, pd, modules['pwscup2026.kit.validate'])
    return np, pd, modules['pwscup2026.kit.validate'], cfg, dist


def infer(destination):
    np, pd, validator, cfg, dist = actual()
    read = lambda p: pd.read_csv(p, float_precision='round_trip')
    result = predict_world(candidates=read(w.ORIGINALS/'B.csv'),
        self_labels=read(w.ORIGINALS/'B_self.csv'), background=read(w.ORIGINALS/'A_bg.csv'),
        self_challenge=read(w.TARGETS/'aia_challenge_10.csv'),
        targets={k: {'release': read(w.TARGETS/f'C_{k}.csv'),
            'challenge': read(w.TARGETS/f'aia_challenge_{k}.csv')} for k in w.ATTACK_TARGETS},
        schema=json.loads((w.ORIGINALS/'schema.json').read_text(encoding='utf-8-sig')), cfg=cfg)
    validation = w.validate_frames(np,pd,validator,result['F_mia'],result['F_aia'],cfg,dist)
    destination.mkdir(parents=True)
    for name in ('F_mia','F_aia'):
        result[name].to_csv(destination/(name+'.csv'),index=False,lineterminator='\n',float_format='%.17g')
    save(destination/'diagnostics.json',result['diagnostics'])
    save(destination/'validation.json',validation)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--infer',type=Path)
    parser.add_argument('--output',type=Path,default=HERE/'submission')
    args=parser.parse_args()
    if args.infer:
        infer(args.infer.resolve()); return
    output=args.output.resolve()
    if not output.is_relative_to(HERE) or output.exists():
        raise ValueError('Use a new output directory inside pl1')
    work=HERE/'.work'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    for i in (1,2):
        subprocess.run([sys.executable,'-B',str(HERE/'run.py'),'--infer',str(work/str(i))],check=True)
    content={name:(work/'1'/name).read_bytes() for name in ('F_mia.csv','F_aia.csv')}
    for name,data in content.items():
        if data != (work/'2'/name).read_bytes():
            raise ValueError('Independent generation differs: '+name)
    np,pd,validator,cfg,dist=actual()
    output.mkdir(parents=True)
    for name,data in content.items(): (output/name).write_bytes(data)
    zipped=output/'team10_pl1_attack.zip'
    with zipfile.ZipFile(zipped,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in content.items(): z.writestr(name,data)
        z.writestr('token.txt',(w.ORIGINALS/'token.txt').read_bytes())
    check=validator.validate_submission_zip(zipped,cfg,w.TARGETS,'attack')
    if not check.ok: raise ValueError('Official ZIP validation failed: '+str(check.errors))
    with zipfile.ZipFile(zipped) as z:
        if z.namelist()!=['F_mia.csv','F_aia.csv','token.txt']: raise ValueError('Invalid ZIP members')
        if z.read('token.txt')!=(w.ORIGINALS/'token.txt').read_bytes(): raise ValueError('Token mismatch')
    frames={n:pd.read_csv(output/n,float_precision='round_trip') for n in content}
    validation=w.validate_frames(np,pd,validator,frames['F_mia.csv'],frames['F_aia.csv'],cfg,dist)
    save(output/'verification.json',{'model':'attackT/pl1','created_utc':datetime.now(timezone.utc).isoformat(),
        'two_processes_byte_identical':True,'official_zip_valid':True,'actual_id_validation':validation,
        'original_token_bytes_match':True,'official_submission_performed':False,'official_score_measured':False,
        'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [*(output/n for n in content),zipped]}})
    (output/'diagnostics.json').write_bytes((work/'1'/'diagnostics.json').read_bytes())
    print('Validated local pl1 submission: '+str(zipped),flush=True)


if __name__=='__main__': main()
