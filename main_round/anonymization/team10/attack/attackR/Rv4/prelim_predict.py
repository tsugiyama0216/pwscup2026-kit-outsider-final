"""Generate preliminary predictions without reading any target truths."""
import runtime as rt
import pandas as pd
import yaml
from model import variants
def main():
    out=rt.HERE/'evaluation/prelim-predictions'
    if out.exists():raise ValueError('Preserve existing predictions')
    data=rt.MOCK/'mock_TS_using_prelim/data/public';own=data/'team24_processing';tar=data/'attack_targets'
    read=lambda p:pd.read_csv(p,float_precision='round_trip')
    targets=sorted(int(p.stem.split('_')[-1]) for p in tar.glob('C_*.csv') if p.stem!='C_24')
    cfg=yaml.safe_load((rt.MOCK/'mock_TS_using_prelim/official-kit/starter/src/pwscup2026/kit/kit_config.yaml').read_text(encoding='utf-8-sig'))
    cfg.setdefault('aia',{}).update({'qi_continuous':['age','SBP','TG','HDL','ALT'],'block_keys':['sex','prefecture'],'k_nn_onset':5})
    public=dict(candidates=read(own/'B.csv'),self_labels=read(own/'B_self.csv'),background=read(own/'A_bg.csv'),self_challenge=read(tar/'aia_challenge_24.csv'),schema=rt.read(own/'schema.json'),cfg=cfg,targets={k:dict(release=read(tar/f'C_{k}.csv'),challenge=read(tar/f'aia_challenge_{k}.csv')) for k in targets})
    before=rt.preserve();v=variants(**public);out.mkdir(parents=True)
    for comp,n in [('mia','F_mia'),('aia','F_aia')]:
        for name,frame in v[comp].items():
            p=out/comp/name;p.mkdir(parents=True);(p/(n+'.csv')).write_bytes(rt.csv(frame))
    rt.save(out/'provenance.json',{'public_input_sha256':{p.relative_to(data).as_posix():rt.sha(p) for p in data.rglob('*') if p.is_file()},'frozen_dependencies':before,'diagnostics':v['diagnostics'],'uses_historical_T_fitted_assets':True,'target_truth_used':False})
    rt.preserve()
if __name__=='__main__':main()
