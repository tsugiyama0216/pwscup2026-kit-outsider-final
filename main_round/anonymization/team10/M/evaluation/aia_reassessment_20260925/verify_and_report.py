"""Validate provenance and publish aggregate-only local reevaluation results."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess, sys
import numpy as np
import pandas as pd
import evaluate as e

def md_table(rows, columns):
    return '\n'.join(['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']+
                    ['| '+' | '.join(str(row[c]) for c in columns)+' |' for row in rows])

def main():
    table=pd.read_csv(e.HERE/'comparison_all_seeds.csv')
    primary=pd.read_csv(e.HERE/'comparison_primary.csv')
    sensitivity=pd.read_csv(e.HERE/'sensitivity.csv')
    prior=e.read(e.TEAM/'M/M1_v5/protocol.json')
    assert all(e.sha(e.TEAM/'participant_data'/f)==h for f,h in prior['inputs_sha256'].items())
    b=pd.read_csv(e.TEAM/'participant_data/B_self.csv');a=pd.read_csv(e.TEAM/'participant_data/A_bg.csv')
    rare=np.r_[b.is_rare.to_numpy(bool),e.ev.local.rare_mask(a,e.read(e.TEAM/'participant_data/schema.json'))]
    for seed in e.SEEDS:
        plan=e.read(e.CONFIRM/f'cohorts_{seed}.json')
        for attacker in plan['attackers']:
            idx=np.array(attacker['pool_indices'])
            assert np.array_equal(idx<len(b),np.array(attacker['truth'],bool))
            assert np.array_equal(rare[idx],np.array(attacker['rare'],bool))
    run=subprocess.run([sys.executable,'-X','utf8',str(e.HERE/'test_evaluation.py')],capture_output=True,text=True,encoding='utf-8')
    (e.HERE/'logs/test_results.txt').write_text(run.stdout+run.stderr,encoding='utf-8')
    if run.returncode: raise RuntimeError('Focused tests failed: see logs/test_results.txt')
    completion=e.read(e.HERE/'completion.json')
    assert completion['protocol_sha256']==e.sha(e.HERE/'protocol.json')
    assert e.read(e.HERE/'protocol.json')['code_sha256']==e.sha(e.HERE/'evaluate.py')
    assert all(e.sha(e.HERE/f)==h for f,h in completion['result_sha256'].items())
    pivot=table.pivot(index='seed',columns='model',values='Objective_no_phi_proxy')
    dif=pivot['M1_v5']-pivot['M1_v4']
    comparison={'v5_vs_v4_improved':int((dif>1e-12).sum()),'tied':int((np.abs(dif)<=1e-12).sum()),'decreased':int((dif < -1e-12).sum())}
    e.save(e.HERE/'verification.json',{'verified_utc':datetime.now(timezone.utc).isoformat(),
       'tests_passed':5,'test_log':'logs/test_results.txt','current_data_matches_prior_mia_input_hashes':True,
       'all_180_mia_attacker_truth_and_rare_arrays_match_current_data':True,
       'result_hashes_match_completion':True,'protocol_and_code_hashes_match':True,
       'candidate_csv_and_submission_zip_unchanged':True,'comparison':comparison,
       'test_code_sha256':e.sha(e.HERE/'test_evaluation.py'),'report_code_sha256':e.sha(__file__)})
    old=[]
    for model in e.MODELS:
        oldpath=(e.OLD/'results/M1_v3/aia.json' if model=='M1_v3' else e.TEAM/'M'/model/'aia.json')
        old.append({'model':model,'legacy_AIA_full_query':e.read(oldpath)['worst_before_phi'],'source_sha256':e.sha(oldpath)})
    e.save(e.HERE/'legacy_aia_reference.json',old)
    primary_rows=[{'モデル':r.model,'U':f'{r.U:.6f}','a_MIA':f'{r.a_MIA:.6f}',
                   'AIA（φなし）':f'{r.AIA_before_phi:.6f}','Anon代理（φなし）':f'{r.Anon_no_phi_proxy:.6f}',
                   '総合代理':f'{r.Objective_no_phi_proxy:.6f}','φ控除仮定のAnon':f'{r.Anon_phi_scenario:.6f}'} for r in primary.itertuples()]
    sensitivity_rows=[{'モデル':r.model,'Anon最小':f'{r.Anon_no_phi_proxy_min:.6f}','Anon中央値':f'{r.Anon_no_phi_proxy_median:.6f}',
                       'Anon最大':f'{r.Anon_no_phi_proxy_max:.6f}','総合最小':f'{r.Objective_no_phi_proxy_min:.6f}',
                       '総合中央値':f'{r.Objective_no_phi_proxy_median:.6f}'} for r in sensitivity.itertuples()]
    diag=pd.read_csv(e.HERE/'all_B_all_A_QI_only.csv')
    diag_rows=[{'モデル':r.model,'旧AIA・非公開属性を含む':f"{next(x['legacy_AIA_full_query'] for x in old if x['model']==r.model):.6f}",
               '同じ全B/全A・QI限定':f'{r.all_B_all_A_QI_only:.6f}',
               '163対163・主seed':f"{primary.set_index('model').loc[r.model,'AIA_before_phi']:.6f}"} for r in diag.itertuples()]
    report='''# M1_v3・M1_v4・M1_v5の再評価（2026-09-25）

**V5は今回の9条件でもV4に対して総合代理値が5条件で改善、4条件で同点、低下0条件でした。ただし、主条件のAnon代理は旧評価0.930308から0.846154へ修正されます。公式Anonは確定できません。**

ユーザー提供の引き継ぎ資料を受け、固定された3候補の有用性と匿名性代理を再評価しました。正式な提出選択はユーザー指定のM1_v3のままです。候補CSV・提出ZIPは変更していません。

## 主条件の結果

seed=20260924。UとAnonは高いほどよく、a_MIA・AIAは低いほどよい指標です。総合代理=min(U, Anon代理)。φ無控除を基本表示にしますが、これが公式スコアの下限であるという意味ではありません。

PRIMARY_TABLE

φ₁₀=0.026887はユーザー提供値です。実際の非公開challenge用の補正値を今回の疑似challengeに適用できるとは確認していません。最後の列は「仮に同じφを引いたら」という計算例であり、提出予測ではありません。V5の同シナリオの総合代理も0.873041です。

V3とV4はDOMIASが同じで主条件のa_MIA=0.402098、V5では0に下がりました。一方、主条件のAIAは3候補とも0.153846です。V5の主条件ではAIAが匿名性の制約となります。a_MIA=0は、この局所評価で偶然補正後の超過が残らなかった意味で、MIA安全性の保証ではありません。

## 9通りの抽出での変動

SENSITIVITY_TABLE

V5の最低Anonは0.632867（seed=20261012）で、その条件はMIAが制約です。9条件の範囲は信頼区間ではありません。全条件は同じB/Aからの再抽出であり、独立した母集団への外部検証ではありません。V5の選択に使われた既知6seedと、以前の確認3seedを再利用しており、今回新設のホールドアウトではありません。

## AIAの何を直したか

公式ルールブックmain-process-20260912の§4.2・§6.2.2と固定公式コードを照合しました。

- 攻撃入力をage・SBP・TG・HDL・ALT・sex・prefectureの7属性に限定。BMI・FPG・time・onset・is_rare・record_idは予測器に渡しません。正解time/onset/rareは標的選定と採点にのみ使います。
- 標的163人・対照163人。各側に発症者16人、希少群13人、そのうち希少発症者4人を含めます。
- 県・性別・年齢帯・発症の1対1対応に加え、希少構成も対応させます。年齢帯はルール本文で幅を確認できなかったため5歳幅を仮定しました。
- 攻撃は公式QI最近傍と、許可されたQIだけを使う3視点の最近傍推定の中央値の2種類を事前固定。time誤差±0.5年以内を成功とし、全体および希少群の公式採点を使用。希少側のSEデッドゾーンはz=1.64です。2攻撃の最大値を取り、φシナリオではその後に1回だけφを引きます。
- DOMIASは全6,192人の確信度を3候補で再計算し、保存値との完全一致を確認。各seedの真値・希少ラベルとのハッシュ、観測TPR、20,000置換の保存配列の95%点と正規化式を検証して、同じ置換結果を再利用しました。置換20,000回を今回新たに実行したわけではありません。
- 有用性Uは3候補とも固定公式キットで再採点し、以前の値と一致しました。

主seedのV5の多視点AIAは、標的成功130/163、対照成功124/163で、(130−124)/(163−124)=6/39=0.153846です。希少群は両側8/13の成功で希少リスクは0。公式最近傍単独のリスクは0.090909で、多視点側が採点を決めています。

## 旧評価との差を切り分ける

DIAGNOSTIC_TABLE

中央列は人数構成を変えず、公開されるQIだけに限定した診断です。最右列は人数・希少構成・対応抽出も変更しています。これらは評価条件が違うため、横方向の差をモデルの改善と読んではいけません。

**V5がAIAでも一貫して改善するという根拠はありません。** 全B/全AのQI限定診断ではV5のAIAがV4より高く、163対163でもseed=20261003のV4/V5は0.214286、V3は0.195122でした。今回のV5の総合優位は主にDOMIAS低下によるものです。

## 公式評価との違い

本戦AIAの標的は他チームと共有しないPiから選ばれますが、Piの内訳は未公開です。今回はB_selfから抽出しています。本来の対照は全チームのコホートにもA_bgにも含まれませんが、そのデータもありません。今回はA_bgを対照に使っており、生成・背景分布・対照に公開Aを共用する影響があります。

本戦と同じ構成の抽出を可能にするため、公開Aに同じ対応条件を持つ人がいるBの999人から標的を選びました。50人は候補から外れています。希少・発症の適格Bは6/11人、希少・非発症は11/13人であり、選別による偏りを無視できません。A側の希少ラベルも真値でなく公開検出器の推定です。

また、今回のAIA抽出と以前のMIA共有ブロック抽出は別々の代理集団です。両者を合わせて真のPiと全共有ブロックを再現したものではありません。2種類のAIAおよび参照DOMIASは本番の全攻撃を網羅しません。φの正式な適用確認も残っています。そのため、Anon=1の条件でも完全な匿名性は主張できません。

## 保存物・再現

- [主条件CSV](comparison_primary.csv)、[全27条件CSV](comparison_all_seeds.csv)、[抽出感度CSV](sensitivity.csv)
- [全54件のAIA攻撃内訳](aia_attack_details.csv)、[全B/全AのQI限定診断](all_B_all_A_QI_only.csv)
- [固定条件・入力ハッシュ](protocol.json)、[完了記録](completion.json)、[検証記録](verification.json)、[MIA保存結果の照合](mia_cache_verification.json)
- [5テストの結果](logs/test_results.txt)。公開属性以外を欠損にしても予測不変、同条件の抽出再現、人数・対応・重複、閾値と希少SE計算、最大攻撃とφ控除順序、総合値と入力不変を確認しました。

このフォルダーのevaluate.pyを既存のwork/.venvのPythonで実行し、次にverify_and_report.pyを実行すると再現できます。公式キットcommitは3dd71bd3436f2abfdbb64da97e860f22729ea054です。原本ルールブックはwork/official-kit/rulebook内のmain-process-20260912 PDF、参照実装はstarter/src/pwscup2026/attack/aia.py・scoring/attack_score.py・kit/codabench/scoring_io.pyです。

個人単位の抽出位置と予測はprivate/にローカル保存し、集計レポートとは分けてGitの対象から除外しています。初回実行は以前のMIA置換配列の保存先解決で停止しました。既存の正しい保存先を追加した後、同じ入力・seed・攻撃定義で完了しています。初回条件記録はlogs/attempt_01_protocol.jsonに保持しました。
'''
    report=report.replace('PRIMARY_TABLE',md_table(primary_rows,list(primary_rows[0])))
    report=report.replace('SENSITIVITY_TABLE',md_table(sensitivity_rows,list(sensitivity_rows[0])))
    report=report.replace('DIAGNOSTIC_TABLE',md_table(diag_rows,list(diag_rows[0])))
    (e.HERE/'README.md').write_text(report,encoding='utf-8')
    payload={'schemaVersion':1,'items':[
       {'id':'updated-comparison','title':'M1_v3・V4・V5の再評価','queries':[{'id':'same-seed-results','source':{
          'label':'固定候補3種の再計算','files':[{'label':f} for f in ['comparison_primary.csv','comparison_all_seeds.csv','sensitivity.csv','verification.json']],
          'metricDefinitions':[{'label':'Anon代理','definition':'Anon代理は、正規化したDOMIASとAIA最大リスクの大きい方を1から引いた局所評価です。基本表示はφ無控除です。'},
            {'label':'総合代理','definition':'総合代理は有用性UとAnon代理の小さい方です。'}],
          'filters':['モデルは固定されたM1_v3・M1_v4・M1_v5','主seedは20260924','変動の確認は同じデータから9通りの抽出'],
          'caveats':['本戦のchallenge・真値・非公開対照がないため、今回のAnon代理は正式な匿名性得点ではありません。',
            'φ=0.026887は提供済みですが、今回の疑似集団への適用は仮定にすぎません。',
            '9条件は独立したデータでの検証ではなく、範囲も信頼区間ではありません。'],
          'executedAt':completion['completed_utc'],
          'evidenceFlow':[{'kind':'result','title':'同条件比較','detail':'V5はV4より総合代理が5条件で高く、4条件で同点でした。最低AnonはV5が0.632867、V3/V4が0.562937でした。'},
             {'kind':'validation','title':'再計算と保存値の照合','detail':'有用性と全候補のDOMIAS確信度を再計算。以前の20,000置換の結果は入力ハッシュと95%点を照合して再利用。5テストに成功し、元CSVと提出ZIPの不変を確認しました。'}]},
          'columns':list(primary.columns),'rows':primary.to_dict('records')},
          {'id':'sensitivity-results','source':{'label':'9抽出の観測範囲','files':[{'label':'sensitivity.csv'}]},
           'columns':['model','Anon_no_phi_proxy_min','Anon_no_phi_proxy_median','Anon_no_phi_proxy_max'],
           'rows':sensitivity[['model','Anon_no_phi_proxy_min','Anon_no_phi_proxy_median','Anon_no_phi_proxy_max']].to_dict('records')}]},
       {'id':'aia-correction','title':'AIA条件の修正と残る制約','queries':[{'id':'allowed-qi-matched-pairs','source':{
          'label':'固定公式ルールとローカルAIA再評価','files':[{'label':'PWS_Cup2026_ルールブック_main-process-20260912.pdf §4.2・6.2.2'},
            {'label':'protocol.json'},{'label':'aia_attack_details.csv'},{'label':'legacy_aia_reference.json'},{'label':'all_B_all_A_QI_only.csv'}],
          'caveats':['標的の真のPi所属は不明です。A_bgを対照にし、適合する対照がいないBの50人を抽出対象から除外しています。',
             '年齢帯は5歳幅と仮定し、公開Aの希少区分には検出器推定を使っています。',
             'MIAとAIAの代理抽出は、実際の共有・非共有ブロックの同時再現ではありません。'],
          'evidenceFlow':[{'kind':'method','title':'入力と構成','detail':'予測には公開7属性のみ。標的・対照は各163人、各側の希少13人・発症16人・希少発症4人を対応抽出。公式最近傍とQI限定多視点の最大リスクを採用しました。'},
             {'kind':'calculation','title':'V5の主条件','detail':'多視点攻撃の成功は標的130/163・対照124/163。全体超過は6/39=0.153846、希少リスクは0でした。Anon代理は1−0.153846=0.846154、φ控除を仮定した場合は0.873041です。'},
             {'kind':'result','title':'前回評価との関係','detail':'旧V5のAnon代理0.930308から今回主条件の0.846154へ修正します。全B/全AでQIに限定した別診断ではV5のAIAはV4より高く、AIAに対する一貫した改善は確認できません。'}]}}]}
      ]}
    e.save(e.HERE/'sources.json',payload)
    print('5 tests passed; provenance verified; aggregate report and source receipt saved.')

if __name__=='__main__': main()
