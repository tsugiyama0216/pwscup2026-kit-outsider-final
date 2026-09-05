# PWSCUP2026 攻撃v7 最終ローカル評価

> 2026-09-05追記：この文書は本番B/A_bgの受領前の研究記録です。
> その後、本番の地元県private-block構造と練習shadowとの不一致を確認しました。
> 下記の`0.356–0.373`等は本番予測として撤回し、独立試験の信頼区間とも
> 扱いません。v7は未提出・未採点で、現在の実測は`0.3004`です。
> 最新判断は `attack_phase_v8/README.md`。候補CSVとZIPは変更していません。

## Executive Summary

最終提出候補`attack-v7-feedback-stack.zip`を実装し、公式package validatorまで通した。
MIAはv4をbyte-identicalに維持し、AIAだけを二点強化した。第一に、3候補すべてが公式0点だった
7 targetで、3方式の全会一致active行を除く補集合へcensoring-edge予測を置いた。第二に、
target 20のmembership maskを変えず、高信頼event link 7行だけ時刻を復元した。

ローカルの主方向推定は`score_1 = 0.356–0.373`、研究間感度を含む広い範囲は
`0.351–0.374`である。前回実測`0.3004`より強い方向を支持するが、private真値に対する
保証や統計的信頼区間ではない。0.5到達を支持する証拠は得られなかった。

## 推奨候補

- ZIP: `local-output/attack-v7/attack-v7-feedback-stack.zip`
- ZIP SHA-256: `CB2A8889C50B4ACEEB9C96D2A892CA640AF7A0DF0896781021D823C5964635D6`
- MIA SHA-256: `D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B`
- AIA SHA-256: `65AD898E8E24497AA1C415BBCBE14D8360BC35676903A069162F4F20F115EAA0`
- 提出操作: 未実施

## 攻撃設計

### MIA

v4をそのまま維持した。Daybreakのcommon順位とv4のrare順位を合成する案は、公式all集計だけなら
v4比の理論下限`+0.001852`を持ったが、採点対象のtarget別`max(all, rare)`を再現できず、
held-out 12/12で敗北した。よって見かけの集計改善より再現性を優先した。

### AIA zero blocks

target `6, 7, 8, 9, 17, 18, 19`は、v4、independent、routedのすべてで公式scoreが0だった。
各行について3方式が全てactiveなら`11.0`、それ以外なら`9.5`とした。これは
`candidate_active = NOT(v4_active AND independent_active AND routed_active)`である。

従来のrouted完全補集合から541行を追加し、active数はtargetごとに217–274となる。
公式形式はactive数を163へ制限せず、全326行に通常の有限`time_hat`を要求するため、形式上も有効である。

### AIA target 20

公式target別winnerの163行maskを固定した。公開Cとのcopula順位空間で相互最近傍かつmargin上位半分の
event linkだけを信頼し、active mask内の7行を公開event時刻へ置換した。広域適用、4行のmask拡張、
他targetへの適用は下振れまたは追加利益の小ささから棄却した。

## 検証結果

| コンポーネント | split / control | n | target平均差 | 勝率 | p10 | score_1換算 |
|---|---|---:|---:|---:|---:|---:|
| Boolean zero-block | independent validation | 26 | +0.048263 | 76.9% | -0.043660 | +0.016892 |
| Boolean zero-block | fresh locked seeds | 30 | +0.044857 | 76.7% | -0.036399 | +0.015700 |
| Boolean zero-block | all-rows control seeds | 25 | +0.058722 | 68.0% | -0.050627 | +0.020553 |
| target 20 decoder | discovery | 32 | +0.08222 | 100% | +0.02125 | +0.00411 |
| target 20 decoder | held-out validation | 32 | +0.09656 | 100% | +0.03349 | +0.00483 |
| MIA hybrid（棄却） | held-out | 12 | -0.052849 | 0% | -0.089380 | -0.052849 |

all-rows controlは完全補集合比`+0.027118`、v7 ruleは`+0.058722`だったため、active数増加だけでなく
全会一致除外にも選別情報がある。target 20はmechanism-preservingなdiscovery/validationの64/64で
非悪化だった。一方、event時刻を独立shuffleするstressではtarget単位で約`-0.113`となり、
この7セルが主要なモデルリスクである。

## 得点再構成

v6 horizon-edgeの既報方向レンジは`0.335–0.352`である。v7の追加差は別target・別操作なので、
次の二項だけを一回ずつ加える。

```text
Boolean fresh lift = 0.0448569453 × 7 / 20 = 0.0156999309
target 20 lift     = 0.09655842 / 20        = 0.0048279210
v7 main range      = 0.335–0.352 + above   = 0.35553–0.37253
```

探索とheld-outの差を広く反映した感度レンジは`0.351–0.374`である。複数seedの平均差を互いに
足したり、target 20のdiscoveryとvalidationを両方足したりはしていない。

## 不採用案と判断理由

- MIA rare-lock hybrid: held-out平均`-0.052849`、0/12勝。
- zero-block部分tail/k: 平均`+0.000415`、勝率46.2%。
- zero-block全行active: 改善はしたが、locked Boolean ruleより平均`0.031603`低い。
- target 20 mask拡張: 追加のscore_1期待値が約`+0.000351`に留まる。
- target 12 rare-proxy: validation rare平均差`-0.494411`で棄却。
- target 19 rare-proxy棄権: 完全新規8 seed・64 preserving casesの平均は
  `+0.051763/target`（score_1換算`+0.002588`）だが、勝率は`48.44% < 60%`。
  全96 casesでv7 complement baselineが0だったため、非悪化は0 clip床由来であり、
  Boolean gainが正の場合の保護を検証できない。多重選択リスクも踏まえ不採用。

## 品質保証

- v7専用テスト5件、repository全118件がPASS。
- `compileall` PASS。
- MIA 1,049行×20 target、全列tieなし。
- AIA 6,520複合キーが一意で、20 target×326行を完全網羅。
- 全入力manifestとSHA、出力SHA、変更targetを固定gateで検証。
- 公式logical validatorと最終ZIP package validatorがPASS。
- ZIP直下は`F_mia.csv`、`F_aia.csv`、`token.txt`だけ。tokenは表示していない。

## 限界とリスク

shadowはproduction Bの真値ではなく、方向選択の代理である。Boolean ruleのfresh確認でもp10は負で、
特にtarget 19は分散が大きい。target 20 decoderはQIとevent-time順序の保存を仮定する。公式集計だけから
全会一致集合の真の符号を識別することはできず、最悪時保証はない。したがってこの候補は
「現在の証拠で期待値を最大化した一案」であり、0.4または0.5を約束するものではない。

## 再現可能性と証跡

- 生成器: `attack_phase_v7/generate_submission_v7.py`
- 候補診断: `local-output/attack-v7/candidate_diagnostics.json`
- Boolean独立評価: `local-output/attack-v7/aia-official-bayes/final_summary.json`
- fresh確認: `local-output/attack-v7/active-fraction/confirmation_summary.json`
- all-rows負対照: `local-output/attack-v7/active-fraction/all_rows_control_summary.json`
- target 20評価: `local-output/attack-v7/aia-time-decoder/REPORT.md`
- AIA希少proxy棄却評価: `local-output/attack-v7/aia-rare/REPORT.md`
- MIA棄却評価: `local-output/attack-v7/mia-lowfpr/FINAL_REPORT.md`
- 最終独立監査: `local-output/attack-v7/final-audit/REPORT.md`
