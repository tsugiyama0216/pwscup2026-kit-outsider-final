"""Render the aggregate audit into a durable Japanese findings document."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent


def render():
    r=json.loads((HERE/'audit_results.json').read_text(encoding='utf-8'))
    n=r['input_counts']
    text=['# R1〜R9 保存済み合成データのプライバシー監査','',
          '2026-09-18。対象は大会で利用を許可された team10 の B / B_self / 公開 A_bg と、保存済み R1〜R9。生成・提出は行っていません。','',
          '**今回の探索的監査では、R9 の有用性向上に伴って個人への攻撃が急激に強くなったという明確な証拠は得られませんでした。ただし、安全性を示す結果ではありません。** 最近傍攻撃には B と公開対照を区別する弱い信号があり、希少群の低偽陽性率での評価は対照数不足です。さらに R9 は都道府県別人数を正確に公開する仕組みで、通常の個人単位の隣接関係に対する差分プライバシー保証を持ちません。','',
          '## 1. 実装から分かること','',
          '| 系列 | 元データから利用する情報 | 原本コード |','|---|---|---|',
          '| R1・R2 | 性別×喫煙別の人数、経験分位点、順位相関、原因別 Cox。R2 は時間区間件数を保って事象を再配置 | [R1/generate.py:46](../R1/generate.py#L46)、[同:61](../R1/generate.py#L61)、[同:127](../R1/generate.py#L127)、[R2/survival.py:86](../R2/survival.py#L86) |',
          '| R3・R4 | Cox 係数と共分散の較正、条件付き平均・共分散の補正 | [R3/calibration.py:35](../R3/calibration.py#L35)、[R4/joint_transport.py:13](../R4/joint_transport.py#L13) |',
          '| R5 | CART の各葉の条件付き経験分布から列を逐次生成。葉サイズ20は k 匿名性やDPを意味しない。R1〜R4のCSVを使わない別系統 | [R5/sequential.py:9](../R5/sequential.py#L9)、[同:26](../R5/sequential.py#L26)、[同:45](../R5/sequential.py#L45) |',
          '| R6〜R8 | 検出希少群の分位点・潜在共分散を合成データへ移す。R8では希少群発症率と Cox を目標に事象交換 | [R6/copula.py:41](../R6/copula.py#L41)、[同:49](../R6/copula.py#L49)、[R8/model.py:63](../R8/model.py#L63)、[同:106](../R8/model.py#L106) |',
          '| R9 | B全体・検出希少群の分位点、順位相関、Coxスコア・標準誤差を同時に近づける。都道府県人数を正確に一致させる | [R9/aggregate_optimizer.py:62](../R9/aggregate_optimizer.py#L62)、[同:73](../R9/aggregate_optimizer.py#L73)、[同:99](../R9/aggregate_optimizer.py#L99)、[同:124](../R9/aggregate_optimizer.py#L124)、[R9/categorical.py:8](../R9/categorical.py#L8) |','',
          'これらは個人行の対応付けをせずに集計統計を合わせる処理ですが、**集計統計を使うだけではDPになりません**。R9の目的関数の `membership` は希少検出区分の保持を表し、会員推論への保護を意味しません。','',
          '### 正確な人数がDPを満たさない理由','',
          '- R1/R2/R3/R4/R6/R7/R8/R9 は B の性別×喫煙人数と完全一致。R1は群ごとに元の群人数だけ生成し、その後の系列でもその人数が保持されます。R5は一致しません。',
          '- R9 は B の都道府県人数とも完全一致。`categorical.py` は B の `value_counts()` を目標にし、出力の人数一致を検査しています。[生成目標:8行](../R9/categorical.py#L8)、[一致検査:34行](../R9/categorical.py#L34)。',
          '- 個人1人の都道府県だけを変えた2つの隣接データを考えると、公開ヒストグラムは必ず変わります。特定のヒストグラムが出る事象の確率が一方で1、他方で0となるため、この「必ず正確な件数を公開する機構」は有限ε・δ<1のDPを満たせません。これは人数を公開情報として固定しない通常の個人単位隣接関係での議論です。',
          '- この構造的な反例は、特定の人物の再識別に成功したという実測ではありません。公開された集計値と、今回の個人単位攻撃の結果を区別します。','',
          '## 2. 共通の比較条件','',
          f"B は {n['B']} 人。B_self の真の希少ラベルは {n['B_true_rare']} 人、公開検出器が希少と判定するのは {n['B_detected_rare']} 人、両方に属するのは {n['B_true_and_detected_rare']} 人です。**真の希少ラベルと検出希少区分は同じではありません。**",'',
          f"公開 A_bg の {n['A_bg']} 人を固定乱数で、参照・学習 {n['public_reference']} 人、閾値較正 {n['public_calibration']} 人、独立評価 {n['public_controls']} 人に分割しました。Bとの13列完全一致は0人。B/A_bgでは {n['shared_record_ids_NOT_person_matches']} 個の record_id が重複しますが、同じIDの値は一致せず、データセット間の人物IDとして使えません。IDだけでは除外していません。",'',
          '全バージョンで同じ分割・同じ尺度・同じ攻撃設定を使い、標準化と公開ベースラインの学習は参照部分だけで行いました。R1〜R9のいずれにもBとの13列完全一致行はなく、A_bgとの完全一致行もありません。完全一致ゼロは匿名性の保証ではありません。','',
          '**A_bg は公開データであり、対象と交換可能な本戦の秘密ホールドアウトではありません。** 観測する識別信号には集団差・生成器の分布差が混ざり得ます。BとA_bgの単変量分布差や相関差もJSONに保存しています。この監査のAUC/TPRを公式MIA失点、公式匿名性、または総合点へ変換してはいけません。','',
          '## 3. 会員推論の探索的評価','',
          '最近傍距離は、合成データに近い人を会員らしいとするスコアです。相対最近傍は公開参照データへの近さで補正します。別に、合成データ対公開参照の分類器も固定設定で評価しました。AUCの角括弧は400回ブートストラップの95%区間。','',
          '| 版 | 全列最近傍AUC [95%] | 相対最近傍AUC | 合成／公開分類器AUC | 最近傍TPR / 対照FPR |','|---|---|---|---|---|']
    for v,a in r['versions'].items():
        m=a['mia']['nn_full']['overall'];ci=m['auc_bootstrap_95'];ratio=a['mia']['nn_ratio_full']['overall']['auc'];clf=a['mia']['synthetic_public_classifier']['overall']['auc']
        text.append(f"| {v} | {m['auc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {ratio:.3f} | {clf:.3f} | {m['tpr']:.2%} / {m['fpr']:.2%} |")
    m=r['versions']['R9']['mia']['nn_full']['overall'];tci=m['tpr_wilson_95'];fci=m['fpr_wilson_95']
    text+=['',f"閾値は較正用対照のFPRが1%以下になる保守的な順位統計で固定しています。独立評価対照のFPRは必ず1%になるわけではありません。R9のTPRは {m['tpr']:.2%}（Wilson95% {tci[0]:.2%}〜{tci[1]:.2%}）、独立対照FPRは {m['fpr']:.2%}（{fci[0]:.2%}〜{fci[1]:.2%}）です。",'',
           'R8からR9への最近傍AUC上昇は小さく、条件付き区間も重なります。R5の最近傍AUCは他の版より高めですが、別の生成方式であり、この比較だけでCARTの因果的な危険性を断定できません。分類器のAUCはほぼ0.5です。攻撃ごとに結果が違うこと自体が、単一の数値を安全証明に使えない理由です。','',
           '### 希少群の検出力不足','',
           f"検出希少群の閾値較正対照は {n['calibration_detected_rare']} 人、評価対照は {n['control_detected_rare']} 人しかいません。保守的な1%FPR較正では閾値が無限大になり、誰も会員と判定できません。**希少群の低FPR TPRは評価不能であり、0%の成功率や安全という意味ではありません。** 生のJSONの `threshold_infinite=true` がこの状態を表します。",'',
           '| 版 | 検出希少群 最近傍AUC [95%] | 相対最近傍AUC | 1%FPR評価 |','|---|---|---|---|']
    for v,a in r['versions'].items():
        m=a['mia']['nn_full']['detected_rare'];ci=m['auc_bootstrap_95'];q=a['mia']['nn_ratio_full']['detected_rare']['auc']
        text.append(f"| {v} | {m['auc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {q:.3f} | 対照数不足 |")
    text+=['','真の希少ラベルはBの24人にしかありません。真の希少Bと検出希少A_bgの比較はラベル定義が異なる参考診断としてJSONに分け、同じ集団の正式な比較として扱っていません。','',
           '## 4. 時刻の属性推論','',
           '公式スキーマのQI（年齢・SBP・TG・HDL・ALT・性別・都道府県）だけから観測時刻を推定しました。正解は±0.5年以内。ExtraTrees平均回帰、5近傍回帰、さらに実際の正解条件に合わせる非線形の時刻分布分類器を試しました。時刻分類器は0.25年刻みの確率分布と打切りの点質量を推定し、±0.5年区間の確率が最大になる時刻を選びます。対照用の公開ベースラインも同じ方法です。','',
           '| 版 | B全体正解率 | 常に10年とする基準 | B事象あり正解率 | 公開学習モデルのB事象あり正解率 | 事象ありの会員側追加利得−対照側追加利得 [95%] |','|---|---|---|---|---|---|']
    for v,a in r['versions'].items():
        mm=a['aia']['extra_trees_interval_mode'];allm=mm['members']['overall'];ev=mm['members']['any_event'];g=mm['gain_difference_diagnostics']['any_event'];ci=g['bootstrap_95']
        text.append(f"| {v} | {allm['accuracy_half_year']:.2%} | {allm['always_ten_accuracy']:.2%} | {ev['accuracy_half_year']:.2%} | {ev['public_baseline_accuracy']:.2%} | {g['member_gain_minus_control_gain']*100:+.2f}pt [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] |")
    aa=r['versions']['R9']['aia']['extra_trees_interval_mode']['members']['overall']
    text+=['',f"R9の全体正解率 {aa['accuracy_half_year']:.2%} は、常に10年と答えるだけの {aa['always_ten_accuracy']:.2%} と近く、公開データだけで学習したモデルも {aa['public_baseline_accuracy']:.2%} です。したがって、84%台の正解率をそのまま個人情報漏洩の大きさと読めません。事象あり181人で見ると、会員に特有の追加推定力が明確に増えたという証拠は今回の設定では得られていません。",'',
           '平均回帰・近傍回帰の成績、発症・死亡・打切り・検出希少・真の希少別の正解率と誤差もJSONに保存しています。これらも実施した攻撃に対する観測であり、より強い時刻推論攻撃が存在しないという証明ではありません。','',
           '## 5. 系列間の依存と次に必要な評価','',
           'R3→R4→R6→R7→R8→R9 は前の合成CSVを読み、保存済みの行を補正しています。R6ではR4の994行が13列すべて同じまま残り、R9はR8の時刻・発症・死亡を行ごとに保持します。このような複数の公開物は独立した合成標本とみなせません。今回の攻撃は各リリース単独が中心で、全リリースを同時に使う最適な合成攻撃は未評価です。','',
           '- 本戦の交換可能な会員／非会員評価集合で、公開A_bgとの集団差を除いたMIA・AIAを測る。',
           '- 希少群の較正・評価対照を増やし、1%以下のFPRで意味のある検出力と区間を確保する。',
           '- 同じ基礎生成器・同じ乱数・同じ評価条件で、希少分位点較正、Cox較正、件数一致を個別にオン／オフする。本レポートの版間差をその因果効果としない。',
           '- DPが要件なら、分位点・相関・カテゴリ人数・Cox・モデル選択を含む全経路で、隣接関係、感度、乱数、プライバシー会計を明示する。正確な私的集計への後処理を戻さない。','',
           '## 6. 再現性と制約','',
           '7件の計算テストが成功。入力と凍結CSVのSHA-256、依存ライブラリ、設定、全集計結果は [audit_results.json](audit_results.json)。行ごとの予測・近傍対応・識別子・トークンは結果に保存していません。検証用に読む情報と公開する個票を分けています。','',
           '400回ブートストラップの区間は、生成物、攻撃モデル、較正閾値を固定した評価標本の不確実性だけを表します。生成・学習・閾値推定の不確実性、集団差、繰り返し候補選択、多重比較は含みません。区間の重なりだけで差がゼロと証明したり、探索した最大値を公式得点と扱ったりしません。','',
           '実行順: `python -m unittest discover -s . -p test_audit.py -v` → `python audit.py` → `python aia_mode.py` → `python write_japanese_findings.py`。対象データと保存済みR1〜R9が必要です。詳細な比較設計は [PROTOCOL.md](PROTOCOL.md)。','']
    (HERE/'FINDINGS.md').write_text('\n'.join(text),encoding='utf-8')


if __name__=='__main__':
    render()
