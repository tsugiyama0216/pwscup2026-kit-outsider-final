# 攻撃手法の探索

**ここは本戦攻撃フェーズの探索専用フォルダーです。** ①〜④を別々に実装し、既存の予備戦実データ評価環境と本戦疑似環境で比較します。採用済み攻撃・公式提出物とは区別します。

2026-10-08に全4方式と公式参照攻撃を評価しました。今回の固定設定では①が4方式中で最高でした。**[評価結果と考察](RESULTS.md)**、[比較レポート](evidence/prelim_fixed_20261008_v1/report.html)から確認できます。

本戦疑似環境でも4世界の評価を完了しました。**[本戦疑似環境の結果と考察](MAIN_RESULTS.md)**、[本戦疑似比較レポート](evidence/main_mock_fixed_20261008_v1/report.html)を参照してください。

GitHubからも結果を確認できるよう、比較表・標的別得点・固定条件・検証記録を [evidence/](evidence/README.md) に保存しています。元の実行結果は `local-output/` に保持し、共有用コピーのSHA-256と出典を記録しています。

| フォルダー | MIA | AIA |
|---|---|---|
| [explore_1_distance](explore_1_distance/README.md) | QI最近傍距離 | 最近傍行のtime |
| [explore_2_density](explore_2_density/README.md) | 公式DOMIAS KDE密度比 | 近傍10行の幅1年最頻区間 |
| [explore_3_gen_lra](explore_3_gen_lra/README.md) | 固定カーネルGen-LRA | ②と同じ |
| [explore_4_predictor](explore_4_predictor/README.md) | 背景とCのtime予測誤差差分 | 固定決定木のtime回帰 |
| explore_common | 同点を保つスコア変換・共通距離 | 共通近傍・時間区間の処理 |

## 評価を実行する

リポジトリ直下から実行します。必要なライブラリは既存の本戦専用Python環境にあります。

```powershell
& .\main_round\attack\.venv\Scripts\python.exe -I -B `
  .\main_round\attack\exploration\run_explorations.py `
  --name prelim_fixed_20261008_v1 --workers 2
```

公式参照攻撃を含む5方式を同じ公開入力で実行します。非空の出力先は上書きしません。再実行時には `--name` を変えてください。個別実行は `--methods explore_1_distance` のように指定できます。

結果は評価環境の規定に合わせて `main_round/mock_test/local-output/mock_TS_using_prelim/<name>/` へ保存します。

- `report.html` / `SUMMARY.md`: 全方式の比較
- `comparison.csv`: MIA・AIA・合計・公式参照との差・標的別の勝敗件数
- `by_target.csv`: 20標的ごとの内訳
- `protocol.json`: 採点前に保存した設定・全攻撃ソースと評価器のSHA-256・依存バージョン
- `<method>/predictions/`: 回答2CSV、公開入力のハッシュ、攻撃ログ
- `<method>/evaluation/`: 公式採点結果と標的別レポート

攻撃は別プロセスで公開テーブルだけを受け取ります。正解は既存評価器が採点するときだけ使います。相手の正解・公開回答を読み込む攻撃や、得点を見て設定を切り替える処理は含みません。ローカルPythonの完全なOS隔離を保証する仕組みではありません。

## 比較の読み方

評価対象は実際の予備戦team24から自己を除く20標的です。MIAは1,049人、AIAは20×326行。予備戦の固定公式採点を使い、本戦のφ控除は適用しません。合計はMIA力とAIA力の和（0〜2）です。

①のAIAと公式参照、②のMIAと公式参照、②と③のAIAの一致を自動確認します。③は背景Rに1行足す際もカーネルを固定したGen-LRA変種で、著者の公開再fit実装そのものではありません。①②③④はMIAで使う情報も異なります。④は候補の既知timeを残差計算だけに使います。

設定は採点前に固定します。この予備戦問題は既に公開・使用されており、結果を本戦の未知データ性能とみなしません。過去の採用攻撃や評価器の原本は変更しません。

## テスト

```powershell
& .\main_round\attack\.venv\Scripts\python.exe -I -B .\main_round\attack\exploration\test_explorations.py
```

人工小データで数式、同距離・同スコア、窓の成功条件、入力の使い方を確認します。評価器自体の既存テストと公開21提出物のReplayは別の検証です。

## 本戦の疑似環境で評価する

①〜④のソース・設定をそのまま使用し、既存のSugiyama v2生成器と本戦公式採点に接続します。通常・厳しい加工構成（mixed/stress）を2seedずつ、同じ生成元分割holdoutから生成する4世界の固定比較です。手法・設定の再選択は行いません。

```powershell
& .\main_round\attack\.venv\Scripts\python.exe -I -B `
  .\main_round\attack\exploration\run_main_mock.py `
  --name main_mock_fixed_20261008_v1 --workers 2
```

各世界は自己10を除く22標的、MIA候補1,049人、AIA7,172行です。AIA補正値φは1,000回の背景抽出で算出し、その世界の全5方式で共用します。攻撃は別プロセスで公開表・本戦設定・公開定数team10だけを受け取ります。

結果は `main_round/mock_test/local-output/exploration/<name>/` に保存します。非空の出力先は上書きしません。再実行時には `--name` を変えてください。

- `comparison.csv` / `SUMMARY.md` / `report.html`: 4世界を同じ重みで平均した比較
- `by_world.csv` / `by_target.csv`: 世界別・標的別の内訳
- `protocol.json`: 採点前に固定した設定、生成元・攻撃・評価器のハッシュ
- `<world>/case/`: 保存された公開入力と採点用正解
- `<world>/scoring_truth/`: 共有する模擬正解・補正値φ・推定誤差の記録
- `<world>/<method>/`: その世界で再生成した回答、採点結果、攻撃ログ

③は固定カーネルGen-LRA変種です。holdoutは既存の生成元分割名であり、既使用データに由来します。4世界は独立した実人口ではなく、同じ有限データからの疑似生成です。本戦の非公開正解・φ・相手の加工方法を再現したものではないため、公式得点の予測値とは扱いません。
