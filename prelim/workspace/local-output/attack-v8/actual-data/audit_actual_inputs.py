"""Read-only audit of user-supplied actual prelim data. Never reads token."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from attack_phase_v3.generate_submission_v3 import discover_known_private_records

ACTUAL=Path('C:/Users/takut/Downloads/PWSCup2026_prelim_team24_アウトサイダー_outsider')
TARGETS=ROOT/'team-shared/prelim/attack-targets'
OUT=Path(__file__).resolve().parent

def main():
    names=('B.csv','B_self.csv','A_bg.csv','schema.json','utility_ref.json','README.txt')
    hashes={name:hashlib.sha256((ACTUAL/name).read_bytes()).hexdigest().upper() for name in names}
    b=pd.read_csv(ACTUAL/'B.csv')
    bs=pd.read_csv(ACTUAL/'B_self.csv')
    bg=pd.read_csv(ACTUAL/'A_bg.csv')
    selfc=pd.read_csv(TARGETS/'aia_challenge_24.csv')
    private,details=discover_known_private_records(b,selfc)
    baseline=pd.read_csv(ROOT/'submissions/prelim/team24/attack-candidate-v4/F_mia.csv')
    checks={
        'B_rows_1049':len(b)==1049,
        'B_self_equals_B':bs[b.columns].equals(b),
        'B_keys_unique':not b.record_id.duplicated().any(),
        'bg_keys_unique':not bg.record_id.duplicated().any(),
        'B_bg_full_records_disjoint':not bool(pd.MultiIndex.from_frame(b.drop(columns='record_id')).isin(pd.MultiIndex.from_frame(bg.drop(columns='record_id'))).any()),
        'B_no_missing':not b.isna().any().any(),
        'bg_no_missing':not bg.isna().any().any(),
        'B_finite_numeric':bool(np.isfinite(b.select_dtypes(include='number')).all().all()),
        'bg_finite_numeric':bool(np.isfinite(bg.select_dtypes(include='number')).all().all()),
        'B_keys_equal_v4':baseline.record_id.astype(str).tolist()==b.record_id.astype(str).tolist(),
        'own_challenge_matches_163':int(private.sum())==163,
        'eligible_rare_equals_11':int((bs.is_rare.astype(bool)&~private).sum())==11,
    }
    toybg=pd.read_csv(ROOT/'participant_data/A_bg.csv')
    q=['age','SBP','TG','HDL','ALT','sex','prefecture']
    bgindex=pd.MultiIndex.from_frame(bg[q])
    toyindex=pd.MultiIndex.from_frame(toybg[q])
    matches=[]
    for target in (1,2,3,4,5,6,7,8,9,10,12,13,17,18,19,20,22,23,26,29):
        release=pd.read_csv(TARGETS/f'C_{target}.csv')
        idx=pd.MultiIndex.from_frame(release[q])
        matches.append({'target':target,'exact_actual_bg_qi':int(idx.isin(bgindex).sum()),
                        'exact_toy_bg_qi':int(idx.isin(toyindex).sum())})
    schema=json.loads((ACTUAL/'schema.json').read_text(encoding='utf-8'))
    toy_schema=json.loads((ROOT/'participant_data/schema.json').read_text(encoding='utf-8'))
    report={'actual_dir':str(ACTUAL),'input_hashes':hashes,'checks':checks,
        'record_id_scope_note':'record_id is a table-local pseudonym, not a global population key; overlapping ID strings do not imply overlapping people',
        'B_bg_record_id_strings_overlap':len(set(b.record_id)&set(bg.record_id)),
        'rows':{'B':len(b),'B_self':len(bs),'A_bg':len(bg)},
        'rare':{'B_true':int(bs.is_rare.sum()),'known_private':int((bs.is_rare.astype(bool)&private).sum())},
        'known_private_diagnostics':details,
        'actual_vs_retained':{'B_equals_B_practice':b.equals(pd.read_csv(ROOT/'participant_data/B_practice.csv')),
            'A_bg_equal':bg.equals(toybg),'rare_detector_equal':schema['rare_detector']==toy_schema['rare_detector']},
        'release_background_matches':matches,'token_read':False,'network_access':False,
        'source_data_modified':False,'production_candidate_changed':False}
    (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    pd.DataFrame(matches).to_csv(OUT/'release_background_matches.csv',index=False)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not all(checks.values()):
        raise SystemExit('INPUT AUDIT FAILED')

if __name__=='__main__':
    main()
