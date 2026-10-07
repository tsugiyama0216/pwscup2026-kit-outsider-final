# Attack_R v2f

ユーザー提供の[R v1公式結果](../v1/official_feedback_user_supplied_20261001.json)（攻撃力合計0.169387840941965）を受け、杉山さんのT v88の模擬データで候補を選んだ第2版です。保存先と使用する模擬データは、このチャットでユーザー確認済みです。

## 採用した変更

| 成分 | R v1 | R v2f |
|---|---|---|
| MIA | M v2の近傍密度比＋22対象間の中央値補正 | T v88の加工方式別MIAを採用し、対象内の順位へ変換。厳密に判明する非共有163人は0にする。 |
| AIA | T v88 | T v88を維持。独自の時刻補正は調整用データで改善が確認できず不採用。 |

MIAではTとMの順位を混ぜる3候補も試しましたが、T単独より低下しました。AIAでは背景との差を使う時刻推定と、非常に近いQIに限定した補正を試しました。平均点だけでなく加工方式別の悪化と、独立生成の診断結果も確認して選択しています。

Tの学習済みモデルとMの既存実装・選択・記録を変更せずに参照します。MIAの順位変換は同値を保持し、対象内ROCを変えません。非共有163人のマスクは模擬正解を流用せず、本番の自己配布B・B_self・自己challengeの厳密な照合から導きます。

## 杉山さんの模擬データ

[protocol.json](protocol.json) に実験前に固定した条件を記録しています。T v88と同じ `build_case`・`make_release`、自己配布A_bgを模擬人物、自Bを背景参照に使います。

- 調整：dev人物分割、seed 2026100101–2026100103。5主要方式×3抽出=15条件、copy・independentの6診断条件。
- 固定後の確認：T v88の既存seed 971001–971005、5方式=25条件。
- 方式：rounded、jitter、strong_jitter、bootstrap、copula。
- 選択：[selection.json](selection.json)。MIA/AIAごとに調整平均で選び、加工方式平均のT比悪化が0.02を超える候補等は除外。確認結果による再選択は行いません。
- 調整の採用候補：MIA平均0.7948717949、AIA平均0.8550957808（正確な値はselection内のauditを参照）。
- 確認結果：[confirmation.json](confirmation.json)。詳細な集計は[共通評価出力](../../../mock_test/local-output/R-v2f/)に保存。

| 固定25条件の確認 | 得点 |
|---|---:|
| v2fのMIA | 0.7307692308 |
| v2fのAIA（φ未控除） | 0.8473837794 |
| 同じ25条件での模擬合計 | 1.5781530102 |

MIAのM側単対象診断は0.5661538462、採用したT側は0.7307692308でした。これはR v1の22対象横断処理との完全な比較ではありません。前回の横断模擬比較ではM v2/R v1が0.672570、T v88が0.690090で、今回の選択とも方向は一致しています。T v88の固定ベンチマーク参照値をMIA・AIAとも再現しました。

**この人物分割と25条件は既に研究で使われており、完全に未知の評価データではありません。** AIAは公式φを引かない模擬得点です。本番の加工法・希少ラベル・配布時の役割も異なるため、R v1の公式0.169388を上回る保証ではありません。R v1は22対象の横断MIAを使うため、今回の単対象MIAのM側比較は成分診断であり、R v1全体の再現得点ではありません。

## 提出用データと検証

- [提出ZIP](submission/team10_R_v2f_attack.zip)
- [F_mia.csv](submission/F_mia.csv)：自Bの1,049人×22対象（record_idを含め23列）
- [F_aia.csv](submission/F_aia.csv)：22対象×326問=7,172行
- [verification.json](verification.json)：独立2回の一致、実配布ID・公式ZIP検証、保存ファイルのハッシュ
- [acceptance_checks.json](acceptance_checks.json)：計8テスト、20種類の不正提出拒否、非共有163行の全対象0、トークン非混入確認
- [production_dependencies.json](production_dependencies.json)：提出生成で照合する固定187ファイル

ZIP直下は `F_mia.csv`、`F_aia.csv`、配布原本の `token.txt` の3件だけです。ZIPはGit対象外。トークンの表示・個別ハッシュ化・別の平文ファイルへの複製はしません。R v1の提出物は保持します。

本番22対象を独立2回生成し、MIA・AIA両CSVの完全一致と公式形式検証に成功しました。MIAは全22対象・19,492セルがR v1から変更され、既知非共有163人の3,586セルは0を維持。AIAはR v1とバイト一致です。R v1のCSV・ZIPは作成時ハッシュのまま保持しています。

ルールは固定版main-process-20260912とmain-attack-20260928に従い、競技の配布入力だけを使用します。Tの `time_hat=11` は元の処理のままです（有限・非負で、公式検証器に上限10の制約はありません）。公式提出・試行消費・GitHub送信は行いません。v2fの公式点は未測定です。

## 再現

本戦専用Python環境を使用します。リポジトリ直下で次を実行してください。既存の非空出力先は上書きしません。

```powershell
./main_round/attack/prepare_attack_m.ps1
./main_round/attack/.venv/Scripts/python.exe -B ./main_round/attack/Attack_R/v2f/test_model.py
./main_round/attack/Attack_R/v2f/run.ps1
./main_round/attack/.venv/Scripts/python.exe -B ./main_round/attack/Attack_R/v2f/test_submission.py
```

再生成先を替える場合は `run.py --output <v2f内の空フォルダー>` を使用します。モデル・選択を変更する場合は次版を作成してください。
