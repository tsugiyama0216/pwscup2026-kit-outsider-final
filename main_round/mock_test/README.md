# 本戦の共通模擬テスト

**模擬問題を生成・保存し、MやTSなどの攻撃手法を同じ問題で比較する共通ツールです。** 攻撃実装は `../attack/M/`・`../attack/attack_TS/` に置き、このフォルダーには問題生成と採点をまとめています。

PowerShellでリポジトリ直下から実行します。

```powershell
Set-Location main_round/mock_test
# 専用Python環境がないPCでは最初に実行
..\attack\setup.ps1

# 3抽出・各7加工方式の模擬問題をまとめて保存
.\generate.ps1 -Split dev -Seed 20260927 -Count 3

# 保存した同じ問題をMのv1で評価
.\evaluate.ps1 -CaseDirectory .\local-output\shadow-cases\dev-20260927-n3 -Plugin ..\attack\M\v1\model.py
```

Mのv1は既定の採用方式 `full_reference_aia` で動きます。別の手法を比較するときは `-Plugin` をその実装へ替え、同じ `-CaseDirectory` を指定します。`-Count` の既定は3で、開始seedから連続する抽出を保存します。6に増やすと、希少共有0・1・2人を各2抽出ずつ確認できます。複数の保存問題もカンマ区切りで指定できます。

問題は `local-output/shadow-cases/`、新しい評価結果も `local-output/` 配下に保存します。複数抽出は `suite.json` と抽出ごとのフォルダーにまとめます。各問題の `public/` は攻撃に渡す入力、`private/` は評価器だけが使う正解です。`REPORT.md` と `summary.csv` の平均・最小最大・公式参照との勝敗に加え、`by_case.csv`・`by_style.csv` で特定の問題や加工方式だけに強くないかを確認してください。既存の非空出力先は上書きしません。

| 評価の入口 | 主集計の条件 |
|---|---|
| 上の保存スイートを評価 | 3抽出×5加工方式＝15条件。copy・independentは別集計 |
| `-Count 6` で保存したスイートを評価 | 6抽出×5加工方式＝30条件 |
| `-Count 1` で保存した単一問題を評価 | 1抽出×5加工方式＝5条件 |
| `-CaseDirectory` を省略 | 既定3抽出×5加工方式＝15条件をその場で生成 |
| TS v88の `published_v88.benchmark` | v88専用の固定5抽出×5加工方式＝25条件 |

複数抽出で、一つの問題だけで高得点になる影響を確認します。候補選択はdevで行い、候補を固定してからholdoutで確認してください。抽出条件が違う得点は直接比較できません。主指標はMIA＋AIAのφ未控除代理値であり、本戦の公式得点ではありません。人物分割のholdoutは既使用で、新seedでも完全に未知の人物による検証にはなりません。

詳細な問題構成、採点、プラグインAPI、複数問題の指定は [local_eval/README.md](local_eval/README.md) を参照してください。公式キット・Python環境・配布原本の参照は [../attack/](../attack/README.md) を共用します。既存のTS/Mのモデル・評価記録は元の作業者フォルダーに保持しています。TS v88の不足依存 `prelim_transfer.methods` は2026-09-30に元コードを回収して復旧しました。v88を使う前に、[専用の再現手順](../attack/attack_TS/attack_research/published_v88/README.md)の `check --runtime` と固定25条件benchmarkを確認してください。
