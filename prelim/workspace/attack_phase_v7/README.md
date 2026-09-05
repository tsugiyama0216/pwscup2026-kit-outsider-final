# 攻撃バージョン7

> 2026-09-05・本番入力受領後の訂正：v7は未提出・未採点です。
> 以下の旧shadow評価は、練習世界と本番の地元県private-block構造の差を
> 十分に再現していません。`0.356–0.373`等の数値を本番得点の予測として
> 扱うことは撤回します。seedが異なる観測も、同じ実データを再利用するため
> 独立した実世界の試験ではありません。現状の実測は`0.3004`です。
> 最新の検証判断は `attack_phase_v8/README.md` を参照してください。
> 既存の候補CSV/ZIPは保存していますが、本番改善を確認したものではありません。

v7は、残り1回の提出に向けた最終の通常形式攻撃です。直近の公式feedbackで得た
target別AIA勝者を土台にし、公式で3方式とも0点だったblockだけを再探索しました。
候補生成はtokenを読まず、ネットワーク接続やCodaBench提出も行いません。

## 最終判断

- MIAは実測で最も堅いv4をbyte-identicalに維持する。
- AIAの13個の既知正点blockはv6 horizon-edge版を土台とし、12 blockはそのまま維持する。
- 公式0点だったtarget `6, 7, 8, 9, 17, 18, 19`では、v4・independent・routedの
  3方式が全てactiveとした行だけ棄権し、それ以外を`time_hat=9.5`とする。
- target 20だけ、既存の163行membership maskを固定したまま、公開表との高信頼な
  相互近傍event link 7行へ時刻を復元する。

提出用ZIPは`local-output/attack-v7/attack-v7-feedback-stack.zip`です。

## v6 horizon-edgeからの変更

| target | v7 active行 | routed完全補集合からの追加 | 時刻復元 |
|---:|---:|---:|---:|
| 6 | 217 | 54 | 0 |
| 7 | 253 | 90 | 0 |
| 8 | 211 | 48 | 0 |
| 9 | 274 | 111 | 0 |
| 17 | 223 | 60 | 0 |
| 18 | 236 | 73 | 0 |
| 19 | 268 | 105 | 0 |
| 20 | 163 | 0 | 7 |

0点7 blockでは合計541行を追加し、削除はありません。target 20で変更する
`challenge_row_id`は`37, 102, 104, 114, 243, 255, 290`です。

## ローカル証拠

| 検証 | shadow観測数 | target-level平均差 | 勝率 | p10 | 機械換算（LB予測でない） |
|---|---:|---:|---:|---:|---:|
| Boolean rule・独立validation | 26 | +0.048263 | 76.9% | -0.043660 | +0.016892 |
| Boolean rule・fresh lock確認 | 30 | +0.044857 | 76.7% | -0.036399 | +0.015700 |
| Boolean rule・all-rows負対照 | 25 | +0.058722 | 68.0% | -0.050627 | +0.020553 |
| target 20・discovery | 32 | +0.08222 | 100% | +0.02125 | +0.00411 |
| target 20・held-out validation | 32 | +0.09656 | 100% | +0.03349 | +0.00483 |

all-rows負対照そのものは完全補集合比`+0.027118`で、v7 ruleはさらに
`+0.031603`でした。したがって改善は単にactive件数を増やした結果ではなく、
3方式の全会一致行を除外することにも情報があります。

## 旧得点換算の記録（本番予測として撤回）

v6 horizon-edgeの方向レンジ`0.335–0.352`に、fresh Boolean差
`0.04485695 × 7 / 20 = 0.01569993`と、target 20 held-out差
`0.09655842 / 20 = 0.00482792`を加えると、主推定は
`0.35553–0.37253`、表示上は約`0.356–0.373`です。研究間の感度を広く取ると
`0.351–0.374`です。

これはprivate真値への信頼区間でも下限保証でもありません。0.4は上振れ時にあり得ますが、
0.5を支持するローカル証拠はありません。

## 不採用にした追加案

- Daybreak common + v4 rare-lock MIA: held-out 12/12でv4に敗れ、平均`-0.052849`。
- 部分tail/k最適化: validation平均`+0.000415`、勝率46.2%で再現せず。
- target 20 membership拡張: score_1換算の追加期待値が約`+0.000351`だけ。
- AIA希少proxyのtarget 12案: held-outのrare差が大幅に負。
- target 19 rare-proxy棄権: 完全新規8 seed・64 preserving casesで平均は
  `+0.051763/target`だったが、勝率`48.44% < 60%`。全96 casesでv7 baselineが
  0だったため非悪化は0 clip床に守られただけで、Boolean gainを保護できる証拠がない。
  最終ZIPには含めない。

## 検証と固定値

- v7専用unit test: 5/5 PASS
- repository全体: 118/118 PASS
- `compileall`: PASS
- 公式logical validator: PASS
- 公式ZIP package validator: PASS
- ZIP SHA-256: `CB2A8889C50B4ACEEB9C96D2A892CA640AF7A0DF0896781021D823C5964635D6`
- `F_mia.csv`: `D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B`
- `F_aia.csv`: `65AD898E8E24497AA1C415BBCBE14D8360BC35676903A069162F4F20F115EAA0`

ZIPは直下に`F_mia.csv`、`F_aia.csv`、`token.txt`だけを持ちます。token内容は表示せず、
候補ディレクトリには保存していません。

## 再現

```powershell
$python = "C:\Users\takut\.codex\.chatgpt-projects\g-p-6a853f607498819195a7ef70756f71c9\pwscup2026-kit-private\starter\.venv\Scripts\python.exe"
& $python -m attack_phase_v7.generate_submission_v7
& $python -m unittest discover -s attack_phase_v7/tests -v
& $python -m unittest discover -v
& $python -m attack_phase_v5.package_submission_v5 `
  --candidate-dir .\submissions\prelim\team24\attack-candidate-v7-feedback-stack `
  --out .\local-output\attack-v7\attack-v7-feedback-stack.zip
```
