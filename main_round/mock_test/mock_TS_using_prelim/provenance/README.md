# データの出典と再配置の記録

2026-10-07の追加依頼で、このデータ一式は既存の非公開GitHubへの保存対象になりました。`data_manifest.json` の `local_only: true` はコピー作成時の履歴として維持します。保存先・公開範囲・Git原本での検証は [publication_20261007.json](../publication_20261007.json) に別記します。

[data_manifest.json](data_manifest.json) は `mock_TS_using_prelim` フォルダーを基準にした178データファイルの台帳です。`files[].path` がコピー先、`sha256` と `bytes` が保存物の照合値です。元ファイルの絶対パス・リポジトリ内パス・公開元URL・元SHA-256、用途、コピー理由、CSV行数・列名も記録しています。

原本コピーは `is_byte_identical_copy: true`、ZIP展開・Excel保存値CSVは `false` とし、`derivation` に元ZIPメンバー、元Excelシート、列名対応などを残しています。ZIP内CSVのバイト列は無改変です。Excel自体は書き換えず、保存値を標準ライブラリのzip/XML読取りで抽出しました。

## 使用した資料

| 資料 | コピー元・公開元 |
|---|---|
| 自チーム加工配布物 | ユーザーのDownloads内 `PWSCup2026_prelim_team24_アウトサイダー_outsider`。4主要ファイルは当時の固定SHA-256を検証し、練習データの混入を防止 |
| 攻撃配布物 | Downloads内 `PWSCup2026_prelim_attack`。[公式予備戦攻撃リリース](https://github.com/pwscup/pwscup2026-kit/releases/tag/prelim-attack-20260825)。配布SHA256SUMSの44件を照合 |
| 正解 | `prelim/results/20260909/sources/ground_truth/`。[運営公開の正解](https://drive.google.com/file/d/1jnYGBEX1vjSgrvzBywHjeBnYru9_H5MQ/view) |
| 全チームの掲載回答 | `prelim/results/20260909/sources/attack_submissions/`。[運営公開の回答](https://drive.google.com/file/d/1x1KGX_XmEMunHqEmjq8ZYMcDqx5BtQs8/view) |
| 公式結果Excel | `prelim/results/20260909/sources/公開用結果表_20260909.xlsx`。[運営公開結果表](https://drive.google.com/file/d/1BLiwflR7FJ9tp5lCeSkP-kJga4ByHRnB/view) |
| 既存の集計と照合記録 | `prelim/results/20260909/` のREADME、manifest、2 CSVを保存。これらは既存のローカル解析資料 |
| 自チーム最終提出 | `prelim/anonymization/anonymization-submitted-v5/C.csv`、`prelim/attack/final/`、`prelim/attack/results/guarded_final.json` |

元Downloadsのチーム名には結合濁点が含まれます。再配置時に名前を推測せず、manifestに記録した絶対パスかPowerShellのファイル選択結果を渡してください。

## 再配置スクリプト

[copy_data.py](copy_data.py) はデータコピーと台帳生成のための補助スクリプトです。通常の評価では再実行は不要で、評価コードは保存済み `data/` を使用します。標準ライブラリだけで実行できます。

リポジトリ直下からの例：

```powershell
$processing = Get-ChildItem -LiteralPath "$env:USERPROFILE\Downloads" -Directory |
    Where-Object Name -like 'PWSCup2026_prelim_team24_*'
if (@($processing).Count -ne 1) { throw '実team24配布フォルダーを1件に特定してください' }
.\main_round\attack\.venv\Scripts\python.exe `
    .\main_round\mock_test\mock_TS_using_prelim\provenance\copy_data.py `
    --processing-directory $processing.FullName `
    --attack-directory "$env:USERPROFILE\Downloads\PWSCup2026_prelim_attack"
```

読み取る加工配布ファイルは6件の許可リストで固定し、トークンは対象外です。既存のコピー先が異なるバイト列なら上書きせず失敗します。実データがないときに `participant_data/` の練習データへ切り替えません。

## 検証範囲

manifestの `verification` に、原本ハッシュ照合、CSVの構造、ID対応、自己提出との一致を保存しています。公表ExcelにあるAIA明細は419組で、攻撃者4→標的23の欠落を維持しています。コホート21の公表点空欄も維持しています。

この台帳はデータ保存の検証です。公式採点との数値照合や攻撃プラグイン実行結果は、評価器側の検証記録を参照してください。予備戦実問題に対する採点であり、本戦の隠れた正解・控除値・匿名化分布を再現したという主張はしません。
