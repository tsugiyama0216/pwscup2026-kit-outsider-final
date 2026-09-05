# AIA rare-channel strengthening audit

## 結論

**希少チャネル向けの追加変更はすべて不採用とし、現在の v7 `F_aia.csv` を維持する。**

最終候補だった target 19 限定ルール（公開特徴から推定した希少候補上位 26 行を `time_hat=11.0` にする）は、完全新規 8 seed の preserving 評価で平均差は正だったものの、勝率が事前基準 60% に届かず 48.44% だった。また、全 96 shadow case で v7 complement baseline の attack power が 0 だったため、baseline が正のケースで既存 Boolean gain を壊さないことを一度も確認できなかった。この正の平均差はゼロ床からの改善だけであり、最終提出に持ち込む根拠として不十分である。

production candidate、ZIP、token、提出状態は変更していない。

## 最終 target 19 監査

評価ルールは次のとおり。

- 対象: target 19 のみ
- 希少候補: `A_bg` で学習した ExtraTrees による公開特徴スコア上位 26 行
- 変更: 該当行を正常形式の棄権値 `time_hat=11.0` に設定
- baseline: v7 の v4 / independent / routed 三方式の unanimous-active の補集合
- preserving: `warp` と `warp_noise_0p03`
- stress: `warp_time_shuffled`
- 完全新規 seed: 20270331–20270338、各 seed 4 shadow target

| 評価集合 | cases | 平均 power 差 | 勝率 | p10 | 最小差 | all 平均差 | rare 平均差 | score_1 換算平均差* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| preserving 全 world | 64 | +0.051763 | 48.44% | 0.000000 | 0.000000 | 0.000000 | +0.051763 | +0.002588 |
| time-shuffle 全 world | 32 | +0.048631 | 46.88% | 0.000000 | 0.000000 | 0.000000 | +0.048631 | +0.002432 |
| preserving baseline>0 | 0 | 評価不能 | 評価不能 | 評価不能 | 評価不能 | 評価不能 | 評価不能 | 評価不能 |
| time-shuffle baseline>0 | 0 | 評価不能 | 評価不能 | 評価不能 | 評価不能 | 評価不能 | 評価不能 | 評価不能 |

\* 20 target の単純平均を仮定した参考換算。公式 score_1 の保証値ではない。

下振れは観測上 0 だったが、これはすべての baseline power が 0 でスコアがゼロ床に張り付いた評価である。したがって「非悪化」は既存の正の attack power を保護した証拠ではない。むしろ baseline-positive subset が空であること自体を重大な evidence gap と判断した。

### 事前ゲート

| 条件 | 結果 | 判定 |
|---|---:|---|
| preserving mean > 0 | +0.051763 | pass |
| preserving win rate >= 0.60 | 0.484375 | **fail** |
| preserving p10 >= 0 | 0.000000 | pass（ゼロ床に注意） |
| all mean >= 0 | 0.000000 | pass |
| stress mean >= 0 | +0.048631 | pass |
| baseline-positive preserving mean/p10 >= 0 | 0 cases | **fail / 未確認** |
| baseline-positive stress mean >= 0 | 0 cases | **fail / 未確認** |
| normal-format | 完全な CSV、有限値 | pass |

総合判定は **reject**。

## production への影響監査

仮に target 19 ルールを production に適用すると、上位 26 行のうち 25 行が現在の v7 active 行と重なり、実際に 25 行の `time_hat` が変わる。残る 1 行は既に v7 complement で inactive/unanimous 側にあり、値は変わらない。したがってこれは小さな補正ではなく、target 19 の active mask を 25 行分変更するルールである。

変更対象の `challenge_row_id` は次の 25 行。

`3, 35, 55, 61, 71, 101, 102, 103, 105, 107, 117, 127, 159, 164, 186, 195, 203, 233, 246, 272, 285, 287, 312, 320, 324`

## 希少 proxy の妥当性と限界

公開 rare detector は `age, BMI, SBP, TG, HDL, ALT, FPG` の 7 変数を使う一方、AIA challenge で見える数値特徴は `age, SBP, TG, HDL, ALT` の 5 変数だけで、`BMI` と `FPG` が欠落する。このため真の rare 判定は復元できず、公開情報だけから proxy を作った。

独立な `B_self` の真 rare ラベルに対して、採用した ExtraTrees proxy は ROC AUC 0.847033、真の陽性数と同数の top-24 で recall / precision 0.291667 だった。順位情報はあるが、上位集合の純度は低く、production で 26 行を一括変更するには不確実性が大きい。

## 広域探索の結果

targets 5, 6, 7, 9, 12, 17, 18, 19, 23 に対して、link quantile、active/edge activation、top-k 拡張、abstention を discovery / validation / time-shuffle で比較した。rare 目的と all 保護を同時に満たす discovery 候補は残らなかった。

とくに一度 power-only 集約で候補に見えた target 12 `activate_edge_k26` は、validation 16 cases で power 平均差 -0.001868、p10 -0.033557、rare 平均差 -0.494411 だった。all 平均差 +0.005630 が `max(all, rare)` を部分的に支えただけであり、rare 強化案として明確に不採用である。

## 再現方法

リポジトリ root から、同じ仮想環境で以下を実行する。

```powershell
$env:PYTHONHASHSEED='0'
& 'C:\Users\takut\.codex\.chatgpt-projects\g-p-6a853f607498819195a7ef70756f71c9\pwscup2026-kit-private\starter\.venv\Scripts\python.exe' 'local-output/attack-v7/aia-rare/evaluate_rare_proxy.py'
& 'C:\Users\takut\.codex\.chatgpt-projects\g-p-6a853f607498819195a7ef70756f71c9\pwscup2026-kit-private\starter\.venv\Scripts\python.exe' 'local-output/attack-v7/aia-rare/inventory_production_rare.py'
& 'C:\Users\takut\.codex\.chatgpt-projects\g-p-6a853f607498819195a7ef70756f71c9\pwscup2026-kit-private\starter\.venv\Scripts\python.exe' 'local-output/attack-v7/aia-rare/benchmark_rare_gate.py'
& 'C:\Users\takut\.codex\.chatgpt-projects\g-p-6a853f607498819195a7ef70756f71c9\pwscup2026-kit-private\starter\.venv\Scripts\python.exe' 'local-output/attack-v7/aia-rare/finalize_rare_gate.py'
& 'C:\Users\takut\.codex\.chatgpt-projects\g-p-6a853f607498819195a7ef70756f71c9\pwscup2026-kit-private\starter\.venv\Scripts\python.exe' 'local-output/attack-v7/aia-rare/audit_target19_abstain.py'
& 'C:\Users\takut\.codex\.chatgpt-projects\g-p-6a853f607498819195a7ef70756f71c9\pwscup2026-kit-private\starter\.venv\Scripts\python.exe' 'local-output/attack-v7/aia-rare/benchmark_target19_fresh.py'
```

主要な機械可読成果物:

- `rare_proxy_validation.json`: proxy の OOF / external validation
- `rare_gate_report.json` と `final_rare_gate.json`: 広域候補探索と rare-specific gate
- `target19_abstain_audit.json`: production で変わる行の監査
- `target19_fresh_report.json`: 完全新規 8 seed の最終判定
- `target19_fresh_scores.csv`: case 単位の paired 差

## 安全境界

使用したのはローカル公開データとローカル shadow evaluator のみである。ネットワークアクセス、token 読み取り、ZIP 作成、production `F_aia.csv` の編集、提出操作は行っていない。
