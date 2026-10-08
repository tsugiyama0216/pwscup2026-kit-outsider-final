# ④ SyntheticPredictor：timeの予測誤差差分

合成データから学んだ `time` 予測器をMIAとAIAで共用する、単一方式の比較実装です。手法・設定を事前固定し、対象別の方式選択やアンサンブルは行いません。

## 手法

背景データ `A_bg` から予測器 `f_bg`、各標的の公開合成データ `C` から予測器 `f_C` を作ります。入力はAIA challengeでも公開されている7列だけです。

- 連続値：`age, SBP, TG, HDL, ALT`。そのまま決定木へ渡します。
- カテゴリ：`sex, prefecture`。背景だけで学習した同一のOneHotEncoderを、背景・各C・候補・challengeに共用します。未知カテゴリは全ゼロです。
- 予測器：`DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=20261008)`。目的変数は学習元テーブルの `time` です。

候補 `x` の既知の `time=t` に対し、MIAの生スコアは次式です。

```text
s(x) = |f_bg(QI(x)) - t| - |f_C(QI(x)) - t|
```

背景モデルよりCモデルが `time` をよく当てる候補ほど、高いスコアを付けます。提出値は共通の `score_to_confidence` で順位を保って0～1に変換します。AIAは同じ `f_C(challenge_QI)` をそのまま `time_hat` とします。

候補Bの `time` は残差の計算だけに使い、予測器や前処理の学習には使いません。BのBMI・FPG・smoking・onset・death、self_labels、self_challenge、self_release、utility_referenceも使いません。対象の非公開正解・過去の提出予測・評価器のファイルをプラグインから読む処理はありません。MIAとAIAは別名簿に対して推論し、行の対応や共有者ラベルを仮定しません。

`config.json` が実行設定です。`predict_world(**public_tables)` は `F_mia`、`F_aia`、`diagnostics` を返します。自チームの標的は除きます。入力の非有限値や不正な `time` は明示的に拒否します。

## 限界と比較

この予測器は二乗誤差を小さくする決定木回帰であり、AIAの±0.5年の成功率や会員・対照の成功率差を直接最適化しません。打ち切り時刻付近の多い集団では、両モデルの残差が同時に小さくなる場合があります。実際の攻撃力は予備戦の公開正解を扱う別プロセスの評価器で測ります。ここには評価用正解を渡しません。予備戦の結果を本戦公式得点の予測とは扱いません。

## 検証

リポジトリ直下から、既存の本戦Python環境で実行します。

```powershell
& .\main_round\attack\.venv\Scripts\python.exe -I -B .\main_round\attack\exploration\explore_4_predictor\tests\test_plugin.py
```

テストは人工テーブルで、残差の向き、背景だけでの前処理学習、候補timeからの学習漏洩がないこと、QI以外の属性への不変性、出力ID・再現性を確認します。予備戦の全体評価は共通評価器から実行します。
