"""Compare verified existing results; do not regenerate, retune, or submit models."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
TEAM=HERE.parents[2]
LATEST=TEAM/'M/evaluation/aia_reassessment_20260925'
OLDER=TEAM/'M/evaluation/domias_main_round_20260924'
OUT=HERE
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old_frozen=read(OLDER/'inputs/frozen_inputs.json');frozen=read(LATEST/'protocol.json')
    checked={**old_frozen['team_input_sha256'],**frozen['input_sha256']}
    for rel,h in checked.items(): assert sha(TEAM/rel)==h,rel
    assert sha(OLDER/'evaluate.py')==old_frozen['evaluator_sha256']
    assert sha(OLDER/'protocol.json')==old_frozen['protocol_sha256']
    assert sha(LATEST/'evaluate.py')==frozen['code_sha256']
    for rel,h in read(LATEST/'completion.json')['result_sha256'].items(): assert sha(LATEST/rel)==h
    assert sha(TEAM/'S/S_DPCopula_ep15/C.csv')==sha(TEAM/'S/S_Copula/epsilon15/release/C.csv')
    latest=pd.read_csv(LATEST/'comparison_all_seeds.csv')
    older=pd.read_csv(OLDER/'comparison_all_seeds.csv')
    assert latest.shape[0]==27 and not latest.duplicated(['model','seed']).any()
    assert older.shape[0]==24 and not older.duplicated(['model','seed']).any()
    np.testing.assert_allclose(latest.Objective_no_phi_proxy,np.minimum(latest.U,1-np.maximum(latest.a_MIA,latest.AIA_before_phi)),rtol=0,atol=1e-12)
    for row in older.itertuples():
        result=read(OLDER/f'results/{row.model}/mia_{row.seed}.json')
        raw=np.mean([x['raw'] for x in result['per_attackers']])
        power=np.clip((max(raw-result['chance95']+.01,.01)/.01-1)/11,0,1)
        assert abs(power-row.a_MIA)<1e-12
        score=read(OLDER/f'results/{row.model}/utility.json')
        assert abs(score['score']['utility']-row.U)<1e-12
    # Common MIA subsets really are identical across the earlier and later comparisons.
    for seed in older.seed.unique():
        assert read(OLDER/f'inputs/cohorts_{seed}.json')==read(TEAM/f'M/M1_v5/confirmation/cohorts_{seed}.json')
    summary=[]
    for name,g in latest.groupby('model',sort=False):
        primary=g[g.seed==20260924].iloc[0]
        summary.append({'model':name,'U':float(g.U.iloc[0]),'primary_Anon':float(primary.Anon_no_phi_proxy),
          'primary_Objective':float(primary.Objective_no_phi_proxy),'objective_median_9':float(g.Objective_no_phi_proxy.median()),
          'objective_min_9':float(g.Objective_no_phi_proxy.min()),'objective_max_9':float(g.Objective_no_phi_proxy.max()),
          'objective_mean_9_diagnostic_only':float(g.Objective_no_phi_proxy.mean())})
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'m1_comparison.csv',index=False)
    pivot=latest.pivot(index='seed',columns='model',values='Objective_no_phi_proxy')
    paired=[]
    for candidate,base in [('M1_v5','M1_v4'),('M1_v5','M1_v3'),('M1_v4','M1_v3')]:
        for scope,seeds in [('all9',list(pivot.index)),('previous_confirmation3',[20261011,20261012,20261013])]:
            d=(pivot[candidate]-pivot[base]).loc[seeds]
            paired.append({'candidate':candidate,'base':base,'scope':scope,'improved':int((d>1e-12).sum()),
                           'tied':int((d.abs()<=1e-12).sum()),'worse':int((d < -1e-12).sum())})
    exclusions=[]
    for name,g in older.groupby('model',sort=False):
        if name=='M1_v3':continue
        exclusions.append({'model':name,'U':float(g.U.iloc[0]),'seeds':len(g),
          'a_MIA_min':float(g.a_MIA.min()),'a_MIA_max':float(g.a_MIA.max()),
          'Anon_upper_bound_from_MIA_max':float((1-g.a_MIA).max()),
          'scope':'Earlier common3 MIA sets; upper bound does not reuse obsolete AIA'})
    other=pd.DataFrame(exclusions);other.to_csv(OUT/'other_candidates_mia.csv',index=False)
    selection=read(TEAM/'M/M1_v5/submission/creation_log.json')
    assert sha(TEAM/'M/M1_v5/submission/team10_M1_v5_defense.zip')==selection['ZIP_sha256']
    assert sha(TEAM/'M/M1_v5/C.csv')==selection['C_sha256']
    caveats=['This is an evidence-based selection among compared fixed CSVs, not proof that all possible models are ranked.',
      'Nine resamples share B and public A; six seeds were used during V5 selection and three were subsequent confirmation.',
      'A_bg is shared by generation, attack background, and controls; unseen nonmembers and actual Pi/shared blocks are unavailable.',
      'AIA controls are public proxies; 50 unmatched B rows were excluded, and only two AIA attacks were tested.',
      'Provided phi=0.026887 is not confirmed applicable to these pseudo challenges; recommendation uses no subtraction.',
      'Some R variants and DPCopula eps1/3/10 lack the same evaluation; absence is not a score of zero.',
      'DP guarantees and empirical contest attack scores are different criteria; no DP violation is inferred from these scores.']
    save('verification.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'recommendation':'M1_v5',
      'selection_objective':'min(U,Anon), same observed scenarios, no phi assumption',
      'checked_input_files':len(checked),'all_hashes_match':True,'comparison_arithmetic_verified':True,
      'shared_first3_mia_cohorts_identical':True,'v5_submission_zip_matches_verified_creation_log':True,
      'paired_comparisons':paired,'caveats':caveats,
      'new_attack_runs':False,'candidate_mutation':False,'submission_performed':False,
      'source_sha256':{str(p.relative_to(TEAM)):sha(p) for p in [LATEST/'comparison_all_seeds.csv',OLDER/'comparison_all_seeds.csv']}})
    sources={'schemaVersion':1,'items':[
      {'id':'m1-choice','title':'現時点の提出候補はM1_v5','queries':[{'id':'paired-m1-results','source':{
        'label':'固定候補V3・V4・V5の最新評価','files':[{'label':'aia_reassessment_20260925/comparison_all_seeds.csv'},
          {'label':'model_selection_20260925/m1_comparison.csv'},{'label':'model_selection_20260925/verification.json'}],
        'metricDefinitions':[{'label':'総合代理値','definition':'総合代理値はUとAnon proxyの小さい方です。Anon proxyは1から正規化MIAとAIAの大きい方を引き、今回はφを控除していません。'}],
        'filters':['既存の固定候補3種','9通りの同じ標的・対照条件','主条件seed20260924'],
        'caveats':['9条件は同じデータの再抽出であり、中央値・最低値は本番の期待値や保証下限ではありません。',
          'V5選択に既知6seedを使用し、残る3seedも同じ母集団の確認です。',
          '本戦の非公開対照・Pi・全攻撃は再現できていません。公開Aの共用と50人の抽出対象除外による偏りも残ります。',
          '提供φ=0.026887の適用を仮定せずに比較しています。'],
        'evidenceFlow':[{'kind':'validation','title':'現物との対応','detail':'全入力と結果ハッシュ、27行の総合式を照合。既存V5提出ZIPも作成記録・評価CSVに一致しました。'},
          {'kind':'result','title':'同じ条件での優劣','detail':'V5対V4は9条件で5改善4同点、以前の確認3条件に限ると2改善1同点。V5対V3は9条件すべてで総合代理値が高くなりました。'},
          {'kind':'result','title':'判断','detail':'V5はV4とUが同じで、総合代理値の中央値と最低値も高いため、得点を目的とする現在の提出候補としてV5を推奨します。AIA単独で一貫して優位という結果ではありません。'}]},
        'columns':list(summary.columns),'rows':summary.to_dict('records')}]},
      {'id':'other-candidates','title':'有用性が高い他候補を選ばない理由','queries':[{'id':'common-mia-screen','source':{
        'label':'8候補の同一3条件DOMIAS比較','files':[{'label':'domias_main_round_20260924/comparison_all_seeds.csv'},
          {'label':'domias_main_round_20260924/inputs/frozen_inputs.json'},{'label':'model_selection_20260925/other_candidates_mia.csv'}],
        'metricDefinitions':[{'label':'MIAからのAnon上限','definition':'Anon=1−max(a_MIA,AIA)なので、どのAIA値でもAnonは1−a_MIA以下です。'}],
        'filters':['全候補共通の3つのMIA候補集合','20疑似攻撃者と20,000置換の保存結果'],
        'caveats':['この比較の旧AIAは最新条件と異なるため、旧AIAを使った順位付けは行っていません。',
          '同条件未評価の候補まで劣ると断定していません。',
          'DPCopulaのDP機構の保証を否定するものではなく、今回の経験的攻撃スコアによる提出判断です。'],
        'evidenceFlow':[{'kind':'validation','title':'比較条件の照合','detail':'元CSVの保存ハッシュ、MIA観測値・偶然補正・正規化を確認し、M1系の初めの3条件と候補集合が同一であることを照合しました。'},
          {'kind':'result','title':'MIAが制約となる候補','detail':'R5、S2、R8、R12、DPCopula ε15・20は3条件すべてでa_MIA=1。この代理評価ではAIAを改善してもAnonは0となるため、現時点の提出候補には推奨しません。'}]},
        'columns':['model','U','seeds','a_MIA_min','a_MIA_max','Anon_upper_bound_from_MIA_max'],
        'rows':other.drop(columns='scope').to_dict('records')}]}]}
    save('sources.json',sources)
    print(summary.to_string(index=False));print(json.dumps(paired,ensure_ascii=False));print(other.to_string(index=False))

if __name__=='__main__':main()
