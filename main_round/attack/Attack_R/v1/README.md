# Attack_R v1

ユーザー指定の組合せ：**MIAはM v2、AIAはT v88**。既存モデルを再学習・再選択せずに呼び出す本戦team10用のモデルです。

2026-10-01追記：ユーザーから `OK` の公式採点結果を受領しました。攻撃力合計は **0.169387840941965**。[提供された原値と意味](official_feedback_user_supplied_20261001.json)を保存しています。以下の未提出・未測定の記載と検証JSONは、このエージェントによる作成時点の記録です。

| 項目 | 採用方法 |
|---|---|
| MIA | `Attack_M/v2/mia.py`。近傍密度比の22対象間中央値を強度1.0で差し引き、順位スコア化。自己配布情報から厳密に分かる非共有163人は0。 |
| AIA | `Attak_T/attack_research/v88/predict.py` の `CombinedAttack.predict()` をそのまま実行し、`combined.aia` だけを採用。T側のMIAは不使用。 |
| 対象 | 自分の10を除く22対象：0–9、11–17、22、25–28。 |

Mの固定済み `local_eval` とTの共通評価器の読み込みが混ざらないように、2方式を別プロセスで実行します。MIAの22対象を一度に計算する処理と、AIAの対象内の行順を保持します。

## 提出用ファイル

- [submission/team10_R_v1_attack.zip](submission/team10_R_v1_attack.zip)：提出用ZIP。直下に `F_mia.csv`、`F_aia.csv`、元の `token.txt` の3ファイルだけを格納。
- [submission/F_mia.csv](submission/F_mia.csv)：1,049行、record_idと22対象の23列。
- [submission/F_aia.csv](submission/F_aia.csv)：7,172行（326行×22対象）、target_k・challenge_row_id・time_hatの3列。
- [verification.json](verification.json)：全件2回の生成一致、M v2保存CSVとの一致、公式検証、実配布IDの検証、ファイルハッシュ。
- [acceptance_checks.json](acceptance_checks.json)：組合せ・提出検査の計8テストと20種類の不正データ拒否、平文トークン非混入の確認。
- [dependencies.json](dependencies.json)：実行前後に照合する172ファイルの固定ハッシュ。
- [upstream_runtime_check.json](upstream_runtime_check.json)：T v88の初期化と固定25条件ベンチマークの再現結果。R v1の公式得点ではありません。
- [RULES.md](RULES.md)：公式ルールと検証範囲の対応。

トークンは表示・個別ハッシュ化・平文複製せず、配布原本からローカルZIP内にだけ格納します。ZIPと中間出力はGit管理対象外です。CodaBenchへの送信・公式試行の消費・今回の成果物のGitHub送信は行いません。公式得点は未測定です。

## 再現

リポジトリ直下で実行します。本戦用環境がなければ先に `main_round/attack/setup.ps1` を実行してください。

```powershell
./main_round/attack/.venv/Scripts/python.exe -B ./main_round/attack/Attack_R/v1/test_model.py
./main_round/attack/Attack_R/v1/run.ps1
./main_round/attack/.venv/Scripts/python.exe -B ./main_round/attack/Attack_R/v1/test_submission.py
```

既存提出物は上書きしません。再生成する場合は `prepare_attack_m.ps1` 実行後、同じPythonから `run.py --output <このv1内の空フォルダーの絶対パス>` を指定してください。トークンを含むZIPは、出力先を変えてもリポジトリ共通の `.gitignore` で除外されます。

生成は2方式×独立2回です。M v2の選択・v1保存チェック、T v88の公開マニフェスト130ファイル、配布原本94ファイルも照合します。凍結したコードの変更は次版で行ってください。既存M/Tのモデル・予測・過去記録は変更しません。
