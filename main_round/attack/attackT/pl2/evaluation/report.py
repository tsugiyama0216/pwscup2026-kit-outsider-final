"""Portable human-readable matched comparison, with explicit limitations."""
from pathlib import Path
import sys,json,html
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime as rt

def main():
    out=rt.MOCK/'local-output/mock_TS/pl2_20261008';sel=json.loads((rt.HERE/'selection.json').read_text(encoding='utf-8'))
    frames=[];diags=[]
    for phase in ['development','confirmation']:
        frames.append(pd.read_csv(rt.HERE/'evaluation'/phase/'comparison.csv'))
        for p in sorted((out/phase/'runs').glob('*/*/predictions/diagnostics.json')):
            d=json.loads(p.read_text(encoding='utf-8'));old,new=d['old_confidence'],d['new_confidence']
            diags.append({'phase':phase,'case':p.parents[2].name,'release':p.parents[1].name,
                'T88_route':d['old_route'],'Tv101_route':d['new_route'],
                'T88_unknown':old['unknown'],'Tv101_unknown':new['unknown'],
                'Tv101_validation_accuracy':d['router_validation']['validation_accuracy'],
                'leave_family_out_unknown_recall':float(np.mean([x['unknown_recall'] for x in d['router_validation']['leave_family_out']]))})
    scores=pd.concat(frames,ignore_index=True);dest=rt.HERE/'evaluation'
    scores.to_csv(dest/'all_scores.csv',index=False)
    summary=scores.groupby(['phase','attack'])[['score_1','mia_power','aia_power']].mean()
    summary.to_csv(dest/'comparison_summary.csv')
    byrelease=scores.groupby(['phase','release','attack'])[['score_1','mia_power','aia_power']].mean();byrelease.to_csv(dest/'comparison_by_release.csv')
    routes=pd.DataFrame(diags);routes.to_csv(dest/'routing_diagnostics.csv',index=False)
    pairs=[]
    for phase,part in scores.groupby('phase'):
        pivot=part.pivot(index=['case_id','release'],columns='attack',values='score_1')
        for base in ['Rv2','Rv4','Tv101','Tv102','Tv103','Tv104']:
            delta=pivot.pl2-pivot[base]
            pairs.append({'phase':phase,'baseline':base,'mean_delta':float(delta.mean()),'wins':int((delta>1e-12).sum()),
                          'ties':int((abs(delta)<=1e-12).sum()),'losses':int((delta< -1e-12).sum()),'worst_delta':float(delta.min())})
    paired=pd.DataFrame(pairs);paired.to_csv(dest/'pl2_paired_comparison.csv',index=False)
    note='各段階は同じ12加工データ×3問題、36条件。AIA補正φは問題ごとに1,000回推定し全モデルで共有。mockTSの単一対象診断であり、22対象共同評価でも公式点の予測でもありません。既使用人物・既存Tの学習履歴・代理希少ラベル・合成対照への依存が残ります。確認結果による再選択なし。'
    display=summary.reset_index().replace({'phase':{'development':'調整用','confirmation':'固定後確認'}}).rename(columns={'phase':'段階','attack':'攻撃モデル','score_1':'合計（模擬）','mia_power':'MIA','aia_power':'AIA'})
    summaryhtml=display.to_html(index=False,float_format=lambda x:f'{x:.9f}',escape=True)
    page='<!doctype html><html lang="ja"><meta charset="utf-8"><title>pl2 / Tv101–104 比較</title><style>body{font:16px system-ui;max-width:1300px;margin:40px auto;padding:0 24px;color:#14222c}table{border-collapse:collapse;width:100%;font-size:14px;margin:22px 0}th,td{border:1px solid #cbd5df;padding:8px;text-align:right}th{background:#edf3f7;position:sticky;top:0}p{line-height:1.8}.note{background:#fff3d7;padding:18px}h1{font-size:30px}details{margin:24px 0}</style><h1>pl2 / Tv101〜Tv104 同条件比較</h1>'
    best=summary.loc['confirmation'].score_1.idxmax()
    page+='<p>pl2：MIA='+html.escape(str(sel['mia_blend']))+'、AIA=4モデルの時刻窓合議（'+html.escape(sel['aia_anchor'])+'を重視）。</p><p>固定後確認で平均が最高だったのは <strong>'+html.escape(best)+'</strong> です。pl2が全候補で最良とは確認できませんでした。確認後の再選択は行っていません。</p><p class="note">'+note+'</p>'+summaryhtml
    page+='<h2>pl2との条件別勝敗</h2>'+paired.to_html(index=False,float_format=lambda x:f'{x:.9f}',escape=True)
    page+='<details><summary>加工データ別の比較</summary>'+byrelease.reset_index().to_html(index=False,float_format=lambda x:f'{x:.6f}',escape=True)+'</details>'
    page+='<details><summary>方式判定と未知判定</summary>'+routes.to_html(index=False,escape=True)+'</details></html>'
    (dest/'comparison.html').write_text(page,encoding='utf-8')
    rt.write(dest/'report_summary.json',{'selection':sel,'paired':pairs,'routing_summary':routes.groupby('phase')[['T88_unknown','Tv101_unknown','Tv101_validation_accuracy','leave_family_out_unknown_recall']].mean().to_dict(),'note':note})
    def md_table(df):
        return '| '+' | '.join(map(str,df.columns))+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join(f'{v:.9f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
    text='# pl2：Tv101〜Tv104から錬成した本戦攻撃\n\n2026-10-08作成。4候補を比較し、調整用mockTSでMIA混合18候補とAIA合議4候補から構成を固定。その後の確認用mockTSでは選び直していません。\n\n'
    text+=f"MIAは **{sel['mia_blend'][0]} × {sel['mia_blend'][2]} + {sel['mia_blend'][1]} × {1-sel['mia_blend'][2]}** の順位混合です。AIAは4モデルの予測時刻について±0.5年窓の支持を比べ、**{sel['aia_anchor']}** の票を2倍にします。単独モデルの改名ではなく、新しい組合せです。\n\n"
    confirmed=summary.loc['confirmation'];best=confirmed.score_1.idxmax()
    finding=f"固定後確認では **{best}** の平均が最高でした。pl2={confirmed.loc['pl2','score_1']:.9f}、Rv2={confirmed.loc['Rv2','score_1']:.9f}、Rv4={confirmed.loc['Rv4','score_1']:.9f}。pl2が全候補で最良とは主張しません。確認結果を見た構成変更はしていません。"
    text+='## 比較\n\n'+finding+'\n\n'+md_table(display)+'\n\n'+note+'\n\n[全比較表](evaluation/comparison.html)・[CSV](evaluation/comparison_summary.csv)・[加工データ別](evaluation/comparison_by_release.csv)・[勝敗](evaluation/pl2_paired_comparison.csv)。\n\n'
    text+='## 候補\n\n- Tv101：公開背景から方式判定器を学習し直す。生成方式を丸ごと除外する診断を含む。T88の攻撃専門部品は固定。\n- Tv102：T88の判定確信度が0.60未満または上位差0.15未満なら不明とし、公式参照へ切替・混合。\n- Tv103：性別・県・年齢を合わせたC/背景の5近傍密度比を、T88のMIA順位へ25%混合。\n- Tv104：Tv101の判定器、未知対応、背景補正を組み合わせる。画像の4項目目は評価手順なので、この統合候補と固定後確認に対応させた。\n\n'
    text+='## 実行\n\n本戦用環境は `main_round/attack/setup.ps1` で構築します。全モデルは保存済みTの学習資産に依存します。新しく学習するのは方式判定器のみで、完全なfresh-background-only評価ではありません。\n\n```powershell\n./main_round/attack/attackT/pl2/run.ps1 -Phase development\n./main_round/attack/attackT/pl2/run.ps1 -Phase confirmation\n```\n\n計算済み条件は再利用します。元の記録を上書きする調整はしません。mockTSの単独実行には `pl2/plugin.py`、各候補には `../Tv101/plugin.py` 等を指定できます。大きな問題・正解・予測はignoredの `main_round/mock_test/local-output/mock_TS/pl2_20261008/` にあります。\n\n'
    text+='本番22対象のCSV作成（確認試験後、新しいpl2内の出力先を指定）：\n\n```powershell\n./main_round/attack/attackT/pl2/production/run.ps1 -OutputDirectory ./main_round/attack/attackT/pl2/reproduced\n```\n\n本番では自己公開情報から証明できる非共有163人を0にします。mockTSにはその自己情報がないためマスクを適用しません。ZIP・トークンコピー・CodaBench提出は行いません。pl2の公式得点は未測定です。\n\n'
    text+='## 保存\n\n- PC：このREADMEと同じ `main_round/attack/attackT/pl2/`。\n- GitHub：[private main / pl2](https://github.com/tsugiyama0216/pwscup2026-kit-private/tree/main/main_round/attack/attackT/pl2)。公開完了のコミットは別途publication記録で確認します。\n'
    (rt.HERE/'README.md').write_text(text,encoding='utf-8')
    for name in ['Tv101','Tv102','Tv103','Tv104']:
        (rt.HERE.parent/name/'README.md').write_text(f'# {name}\n\n2026-10-08の独立候補。公開入力APIは `plugin.py`。実装・固定プロトコル・同条件比較は [pl2](../pl2/README.md) に保存。既存T88/M/Rを変更せず、mockTSでRv2・Rv4と比較した。公式点ではありません。\n',encoding='utf-8')
if __name__=='__main__':main()
