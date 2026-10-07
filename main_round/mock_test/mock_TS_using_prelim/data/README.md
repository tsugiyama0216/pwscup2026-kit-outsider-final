# 予備戦の実配布データと公開結果

このフォルダーは、`mock_TS_using_prelim` のために原本をコピーして整理したデータ一式です。予備戦の自チームは **team24（アウトサイダー / outsider）** です。本戦のteam10やスターターキットの練習データと区別します。

データ178ファイル（18,670,707 bytes）を保存しています。原本の無改変コピー126件と、ZIPから無改変で取り出した42 CSV・Excelの保存値から作った10 CSVです。説明ファイルと `.gitignore` はこの件数に含めません。

## 配置と用途

| 保存先 | 件数 | 内容・用途 |
|---|---:|---|
| [public/team24_processing/](public/team24_processing/) | 6 | 当時の加工フェーズに自チームへ配られた `B.csv`、`B_self.csv`、`A_bg.csv`、`schema.json`、`utility_ref.json`、`README.txt` |
| [public/attack_targets/](public/attack_targets/) | 45 | 当時の攻撃フェーズ配布物。匿名化データ `C_<k>.csv` 21件、AIA問題21件、対象一覧、README、SHA-256一覧 |
| [private/ground_truth/](private/ground_truth/) | 44 | 運営公開の正解。`pool_ids` 22件、`aia_truth` 21件、運営README。採点側だけで使用 |
| [reference/attack_submissions/original_zips/](reference/attack_submissions/original_zips/) | 21 | 運営が認証情報を除去して公開した各チームの回答ZIP原本 |
| [reference/attack_submissions/extracted/](reference/attack_submissions/extracted/) | 42 | 上記ZIP内の `F_mia.csv` と `F_aia.csv`。`cohort01` 等のフォルダー別に無改変で展開 |
| [reference/attack_submissions/README_攻撃フェーズの提出物.md](reference/attack_submissions/README_攻撃フェーズの提出物.md) | 1 | 運営の公開範囲・自己列の取り扱いの説明 |
| [reference/published_results/](reference/published_results/) | 15 | 公式結果Excel、既存集計・検証記録4件、全7シートの保存値CSV、ASCII列名の3 CSV |
| [reference/own_submissions/](reference/own_submissions/) | 4 | 自チームの匿名化v5 `C.csv`、最終攻撃v9の2 CSV、当時ユーザーが共有した公式返却結果JSON |

`public` は「当時、自チームの攻撃で使用できた入力」という意味です。インターネットに公開してよいデータという意味ではありません。2026-10-07のアップロード依頼により、この178ファイルは既存の非公開GitHubへの保存対象です。`.gitignore` は台帳のファイルだけを許可しています。`private` と `reference` は採点・照合のための材料で、攻撃関数に渡す入力にはしません。

## 問題と正解の対応

- **MIA**：`B.csv` の `record_id` はteam24内だけのIDです。`pool_ids_24.csv` で母集団の `pool_id` に変換し、標的の `pool_ids_<k>.csv` の集合に入るかで正解を作ります。別コホートへ `record_id` を直接結び付けません。希少ラベルも攻撃者側の `pool_ids_24.csv` を使います。
- **AIA**：`aia_challenge_<k>.csv` と `aia_truth_<k>.csv` の `challenge_row_id` が対応します。各問題326行は、標的163行・対照163行です。公式の成功率差と希少側補正は評価コードで計算します。
- 配布された21コホートのうち自分24を除き、**自チームの攻撃評価は20標的**です。自己の `C_24` と `aia_challenge_24` は当時の公開入力として保存します。
- コホート21は匿名化データを提出しておらず、標的一覧にはありません。攻撃者側の対応表として公開された `pool_ids_21.csv` は保存しています。

## 公表値との照合用CSV

[team_mapping.csv](reference/published_results/team_mapping.csv) は「総括」の22チーム分です。`cohort_id`、`team_name`、`official_attack_power`、`official_attack_rank` などの列を持ち、採点値がないコホート21は空欄を保っています。

[mia_by_target.csv](reference/published_results/mia_by_target.csv) は公表MIA明細420組、[aia_by_target.csv](reference/published_results/aia_by_target.csv) は公表AIA明細419組です。どちらも `attacker` と `target` で対応させます。日本語列名からASCII列名への対応は [manifest](../provenance/data_manifest.json) の各ファイルの `derivation.column_mapping` に記録しています。

[sheets/](reference/published_results/sheets/) にはExcel全7シートの保存値をそのままCSV化しています。数値はXLSX内に保存された表記を維持し、空欄を0に置き換えていません。書式や数式を表すファイルではありません。公式Excel原本もそのまま保持しています。元Excelの数式セルは0件でした。

公開された **攻撃者4→標的23のAIAは回答・結果とも欠落**しています。この1組は補完していません。公表攻撃力と欠落時の分母の問題は [既存の調査記録](reference/published_results/historical_README.md) に記録されています。同記録の相対リンクはコピー元 `prelim/results/20260909/` を基準とします。

## 原本の確認

[出典・SHA-256 manifest](../provenance/data_manifest.json) に全178ファイルのコピー元、公開元URL、用途、サイズ、SHA-256、派生ファイルの生成法を記録しています。

- 実team24配布データ4件を、当時の `actual_inputs.py` に固定されたSHA-256と照合。
- 運営公開資料67件を既存manifestと照合。
- 攻撃配布物のチェックサム44件と、コピー後178件すべてのSHA-256を照合。
- 自チームの `C.csv` と配布 `C_24.csv`、自チームv9の2 CSVと公開 `cohort24` の2 CSVがそれぞれバイト一致。
- `B.csv` と `B_self.csv` の共通14列、候補IDと正解ID、希少ラベル24行の一致を確認。
- 22コホートのpool表のID一意性、21問題のchallenge ID対応、標的・対照の各163行を確認。

参照した非credential原本を保存しており、認証トークンや認証付き提出ZIPは読み取り・コピーしていません。各チームの公開回答はリーダーボード掲載1件だけです。他チームの非公開B/B_selfや全提出履歴は含まれません。
