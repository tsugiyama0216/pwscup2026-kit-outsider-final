# 攻撃バージョン5: 実験結果

実行日: 2026-09-04

## 結論

第1提出候補は`v4 MIA + Daybreak-independent AIA`です。MIAを完全固定して、v4と十分異なるAIAの
target別情報を得ます。第2候補は`tail12 MIA + structured AIA`、第3提出は実測target別winnerの
合成用に残します。

## 公開実ファイルの構造検査

- 20targetすべて326 challenge行、active既知数は163
- 18targetは`(sex, prefecture, 5歳帯)`全層が偶数
- target 10は奇数層8、target 23は奇数層2
- v4と独立AIAのactive overlapはtarget当たり101〜156 / 163
- structured選択で両expertともinactiveから補う行は20target合計37だけ
- cross-target完全一致C行: target 18は145、19は6、29は162
- C内duplicate occurrences: target 6は1、19は45、29は122

## AIA shadow回帰

練習データから20targetのtruth-known worldを4 seeds生成しました。標準shadowのcontrol poolは小さく、
本番と違って5歳帯matchingに多数失敗するため、評価時だけ各controlのsex・prefecture・5歳帯を
memberにexact-matchさせています。真の時刻・検査値は変更していません。

4 seeds × `raw/noise 3%/noise 10%/time-shuffled` の16条件平均です。

| AIA | power | all | rare | p_m | p_c |
|---|---:|---:|---:|---:|---:|
| v4 reference | 0.911017 | 0.905968 | 0.725028 | 0.908915 | 0.018462 |
| independent reference | 0.893130 | 0.890141 | 0.682499 | 0.896722 | 0.029640 |
| structured本命 | **0.919577** | **0.917620** | **0.729944** | **0.919440** | **0.011273** |
| structured − v4 | **+0.008560** | **+0.011652** | **+0.004916** | +0.010525 | -0.007189 |

本命差はrawとtime-shuffledで0、noise 3%でpower平均`+0.007467`、noise 10%で`+0.026773`でした。
全16 seed/scenario比較の最小power差は0です。rare差は一部seedで負ですが、target別max後の
power平均は非悪化でした。

decoder比較では、独立時刻優先`0.919577`、v4時刻優先`0.917918`、event合意のみ個別時刻
`0.917586`、全10年`0.829551`でした。このため独立時刻優先を採用しています。

negative-controlの`independent` scenarioでは本命にも有限標本由来の正値が出るため、これらの値は
リーダーボード得点の予測には使えません。方式間の方向比較だけに使います。

### 未使用seedでの確認

方式選択に使っていない4 seeds（20260905〜08）×4 scenariosの16条件でも、structured − v4は
power平均`+0.008367`、最小`0`、最大`+0.032186`でした。差が生じるnoise 3%/10%の8条件では
8勝0敗、平均`+0.016733`です。学習側の平均`+0.008560`と近く、方式選択の過適合だけでは
説明しにくい結果でした。

公開構成から推定したonset件数、rare件数、その同時整数制約、年齢近接pairの1-of-2制約も
別途評価しましたが、8条件すべてで現行structuredより悪化しました。pair制約は真の
member/control pairを平均83%復元しても、target当たり64〜98件の強い既存判定を反転するため
採用していません。

cross-target完全一致行フィルタは、注入shadowの作り方で結果が分かれました。target 18/29相当の
局所注入では8条件すべて改善（平均`+0.009493`）、全targetへ160行を一様注入した試験では
4条件平均`-0.000962`でした。したがってフィルタ自体を期待上積みには数えません。ただしAIA memberが
各targetの非共有block由来であるという公開構成と整合し、本番回答への影響もtarget 18/29の各4行、
計8行だけなので、現行候補では維持します。

## MIA実ファイル検査

v3の全target共通bottom-163からprivate 163行を復元しました。v3/v4で全20列のpairwise orderが
一致する唯一最大cliqueから、eligible rare 11行も復元しています。

| tail幅 | 元重複枠 | 修正後 | 変更枠 |
|---|---:|---:|---:|
| 12 | 9 | 0 | 9 / 240 |
| 15 | 16 | 0 | 16 / 300 |

top-12を本命にしています。公式真値が無いため、MIA修正の最終採否は第2提出の実測powerで決めます。

追加shadow（2 seeds × `raw/noise 3%/noise 10%/time-shuffled`）では、v4のtop-12/top-15に
cross-target重複が元から無く、tail12/tail15はいずれもv4と完全同一でした。したがって
「無競合時に変化しないこと」とrare非回帰は確認できましたが、本番で解消する9枠の効果量は
このshadowからは推定できません。初回probeにはMIA修正を入れず、第2提出でのみ切り分けます。

## 提出計画

1. v4 MIA + independent AIA
2. tail12 MIA + structured AIA
3. MIAは実測winner、AIAはv4/1/2の`aia_per_target` winnerを326行blockごとに合成

AIA block-wise winner合成は、各元候補よりAIA target平均を悪化させません。第1・第2提出では
総得点と4内訳だけでなく、20targetの`aia_per_target`、`p_m/p_c`、rare側詳細を必ず保存します。
詳細結果を保存できない場合は、第2提出を`v4 MIA + structured AIA`へ切り替え、変更要因をAIAだけに
限定します。
