# 攻撃バージョン4: 分析・実験結果

実行日: 2026-09-03

## 結論

v2の実測は合計`0.28`、MIA全体`0.0867`、MIA希少`0.3333`、AIA全体`0.1171`、
AIA希少`0.1000`でした。v4は、強いMIA希少順位を固定したうえで全体順位を改善し、AIAでは
公開outcomeの破壊が明確な場合だけ時刻decoderを切り替えます。

## MIA shadow回帰

予備戦と同じ1,049候補、20 targets、targetあたり25〜27会員、共有総数508のshadowを公式
`score_mia_power`で評価しました。known-privateも実データと同じrare 13件/common 150件に層別し、
2 cohort seeds × 2 private seeds × 5 positive scenariosの20 runsです。

| 指標 | v3比平均差 | v3比最小差 |
|---|---:|---:|
| MIA power | +0.021704 | +0.013556 |
| all | +0.030126 | +0.015556 |
| rare | 0 | 0 |

シナリオ別のpower差はraw`+0.0168`、noise 3%`+0.0177`、noise 10%`+0.0207`、
time-shuffled`+0.0356`、mixed`+0.0187`でした。DOMIAS込みの追加4条件でも平均`+0.0182`、
最小`+0.0117`です。単独conformalや単純blendよりRRFの最小改善幅が安定し、eligible-only
retransportは約`-0.04`だったため不採用としました。

実ファイルではtarget 18/29だけがA_bg donor判定条件に該当しました。これらを模した4 world
seeds × 2 private seedsの追加試験では、未filter RRFがdonor除去RRFをtarget 18相当で
`0.1963対0.1670`、target 29相当で`0.4733対0.4456`、20target平均で
`0.92759対0.92474`と上回ったため、本命はdonorを残します。

最終MIAは全20列でv3のrare内順位と一致し、確定non-memberのtop 10/20/50混入はすべて0です。

## AIA shadow回帰

A_bg学習のQI-eventモデルで公開Cの整合性を測り、v3のmembership選択を固定したhorizon decoder
を比較しました。2 seeds × 5 scenariosの10 runsで、閾値`0.58`はtime-shuffledだけを全targetで
切り替え、raw、noise 3%、noise 10%、independentでは一度も不要な切替を行いませんでした。

| 指標 | v3 | v4 | 差 |
|---|---:|---:|---:|
| time-shuffled power/all | 0.697562 | 0.837244 | +0.139682 |
| time-shuffled rare | 0.362789 | 0.482375 | +0.119586 |
| 全scenario平均power | 0.700762 | 0.728698 | +0.027936 |

実ファイルではtarget 17のみ切替対象です。active 163行のうち122行はv3でも既に10.0で、残る
41行だけを10.0へ変更しました。全6,520行のactive総数3,260と選択集合はv3から不変です。
target 7のAUCは`0.5825768316`で閾値より上のため、丸めずadaptiveを維持します。

## 最終生成・検査

- MIA: 1,049行 × 21列、SHA-256
  `D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B`
- AIA: 6,520行 × 3列、SHA-256
  `651D754683BCE4B6F22FD7311AB1584B01C2F42FBB73B13996303472411DFA3A`
- v1〜v4の関連unit tests: 合格
- v4 strict preflightと公式MIA/AIA logical validator: 合格
- CSV再読込、schema、有限値、重複、主要範囲の表形式検査: 合格

事務局限定のtarget別真値は手元にないため、公式得点そのものは提出前には算出できません。
shadow結果は手法比較の回帰試験であり、リーダーボード値の推定ではありません。
