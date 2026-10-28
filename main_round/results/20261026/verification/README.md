# 公開された攻撃結果の再採点確認

公式の提出物・正解データを、保存済みの公式キット `main-attack-20260928`（commit `a490490e11a0ec460a5df9fdefb17c72ad459c1e`）で再採点しました。23チームの攻撃力、MIAの506組、AIAの506組が公式Excelと一致しています。数値7,243件の比較で、最大絶対差は `1.1102230246251565e-16`、許容差は `1e-12` でした。

これは公開後のローカルでの照合記録です。公式結果の正本は [原本Excel](../sources/公開用結果表_本戦_20261026.xlsx)です。

## 条件と確認範囲

- 公式 `score_attack` と `score_mia`・`score_aia` を変更せずに使用。実行前に `main_round/attack/source_manifest.json` の94ファイルをSHA-256で照合しています。
- FPRは `0.01`、AIAの許容誤差は `±0.5` 年、希少群の誤差控除は `z=1.64`。3定数は公式Excelの「メタ」シートとも一致しています。
- AIAは公開された `aia_floor.csv` の標的別φを差し引いてから、自己を除く22標的で平均しています。
- MIAは公開された `pool_ids_<k>.csv` で攻撃者の行IDと母集団IDを対応付け、標的ごとに全体・希少群の大きい値を採ります。
- コホート0・3・5・7・11・16・25のMIAには自己列が残っています。原本を変更せず、公式関数の `team_id` で採点から除外しました。
- MIAの行ID、他22標的の集合、AIAの標的別行ID、重複、有限値も確認しています。

アウトサイダー（本戦コホート10）の再計算値は、MIA力 `0.11993006993006995`、φ控除後のAIA力 `0.07673049828462233`、攻撃力の合計 `0.1966605682146923` です。

有用性U、匿名性の集計、最終順位の再計算はこのスクリプトの対象外です。手法の再学習・予測の再生成・調整・CodaBenchへの提出は行っていません。

## ファイル

| ファイル | 内容 |
|---|---|
| [verification_report.json](verification_report.json) | 条件、照合件数、差、入力と読み込んだ公式コードのハッシュ |
| [attack_scores.csv](attack_scores.csv) | 23チームの再計算MIA・AIA・攻撃力と公式値 |
| [per_target_mia.csv](per_target_mia.csv) | 攻撃者・標的別のMIA 506組。未定義の希少TPRは空欄 |
| [per_target_aia.csv](per_target_aia.csv) | 攻撃者・標的別のAIA 506組。φ控除前後を区別 |
| [verify_official_scores.py](verify_official_scores.py) | 再実行用スクリプト |

## 再実行

リポジトリ直下から、本戦用の既存Python環境で実行します。スクリプトは上記の検証結果JSON・CSVだけを更新し、入力原本・展開済みCSV・公式コードを変更しません。

```powershell
& .\main_round\attack\.venv\Scripts\python.exe -X utf8 .\main_round\results\20261026\verification\verify_official_scores.py
```

初回確認環境はPython 3.12.13です。NumPy・pandasの版と使用した公式ソースのハッシュは検証JSONに保存しています。
