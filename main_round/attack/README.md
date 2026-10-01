# 本戦・攻撃フェーズの作業環境（team10）

作業者Mの正本は `Attack_M/` です。Mの固定版を実行するときは `Attack_M/run.ps1 M-v2-test` 等を使用します。既存環境で旧コマンドを使う場合は、先に `./prepare_attack_m.ps1` でGit管理外の互換配置を準備してください。新規環境では `setup.ps1` が準備します。

**TSのv88の不足依存を2026-09-30に復旧しました。** 元の `prelim_transfer.methods` を `attack_TS/prelim_transfer/` に回収し、予備戦の依存コードも検証対象へ追加しました。別PCでは環境構築後に `published_v88.check --runtime` と固定25条件の `benchmark` を実行します。[担当者・エージェント向けの復旧記録と確認手順](../../CODEX_HANDOFF.md#v88_cross_pc_handoff)を参照してください。

**TS本人の攻撃は [attack_TS/](attack_TS/README.md)、作業者Mの攻撃は [Attack_M/](Attack_M/README.md)、共通の模擬問題生成・採点は [../mock_test/](../mock_test/README.md) です。** 共通ツールで複数抽出を保存し、同じ問題に各攻撃を適用できます。このフォルダーの配布物・公式キット・検証・Python環境を共用し、各作業者のモデル・予測・過去の評価記録は元の場所に保持します。

`attack/local_eval/` の3ファイルは、M v1/v2が固定した依存ハッシュを保つための互換スナップショットです。新しい共通評価器は `../mock_test/local_eval/` を使用します。固定された共有コードは変更せず、共通ツールとTSの入口で必要なパスを初期化します。

**2026-09-30：作業者Mの[v2](Attack_M/v2/README.md)を作成・検証。** 県の照合候補が少なく遠い場合に条件を緩めるAIAと、相手間中央値を差し引くMIAを採用しました。固定後の模擬確認でMIA平均.778927→.815411、AIA（φ未控除）.274991→.370155。別実験であり公式得点の予測ではありません。22対象の[予測CSVと検証記録](Attack_M/local-output/v2/predictions/)を保存し、公式形式・2回生成のバイト一致を確認。v1は保存し、公式提出・ZIP作成はしていません。v2は`main`へ統合し、固定依存とTSの配置の両立を[統合検証記録](Attack_M/main_integration_20260930.json)にまとめています。入口は`attack.ps1 M-v2-test / M-v2-predict`です。

**2026-09-30：作業者Mの[v1の診断・実装・検証結果](Attack_M/v1/README.md)。** [Mの作業一覧](Attack_M/README.md)にコード・予測CSV・検証記録をまとめています。`M`は作業者、`v1`は版名です。他22対象の予測CSVは形式と再現性を確認済みで、公式提出はしていません。実行入口は`attack.ps1 M-v1-diagnose / M-v1-test / M-v1-evaluate / M-v1-predict`です。

2026-09-27の[事務局メール](source_email_20260927.txt)に基づく、このPCの本戦用作業場所です。チームはアウトサイダー / outsider、コホート番号は **10**。予備戦team24とは別です。

**提出期間：2026年9月28日（月）09:00〜10月11日（日）09:00（日本時間）。採点に通る提出は最大8回。** 9月27日の追加依頼に基づき、この環境の配布データ・コード・手順・検証済み参照出力を既存の非公開GitHubリポジトリに保存します。CodaBenchへの提出は行っていません。

**このPCでも2026-09-30に本戦専用環境を構築し、原本94件の照合と公式指定29ライブラリの版を確認しました。** 元の作業PCでは、22対象の公式参照攻撃でMIA 1,049行・AIA 7,172行を生成し、公式検証とCSV再読み込み検証を通過しています。誤った行ID・標的不足・重複・無限値・小数ID・誤トークンを拒否する確認も当時完了しています。仮想環境はGitに含まれないため、別PCでは以下のセットアップが必要です。TS v88は同日の依存復旧後、このPCで初期化に成功しました。固定25条件の再現確認は[復旧記録](attack_TS/attack_research/published_v88/dependency_recovery_20260930.json)を参照してください。

## 使い始める

PowerShellでこのフォルダーを開きます。

```powershell
# リポジトリ直下から
Set-Location main_round/attack

# 配布物・本戦データ・ライブラリ・認証ファイルの存在と形式を確認
.\attack.ps1 check

# 公式参照攻撃を本戦22対象で実行し、生成CSVを検証
.\attack.ps1 baseline

# 将来、提出ZIPができたらローカルで検証
.\attack.ps1 validate .\local-output\候補名\attack.zip
```

専用の仮想環境がないPCや再構築が必要な場合は、このフォルダーで `./setup.ps1` を実行します。Python 3.11以上が必要で、再現用には元PCで使用した3.12.13を推奨します。既存の予備戦用Python・uv、またはPATH上の環境を利用し、本戦専用 `.venv/` に公式 `requirements-kit.txt` の29ライブラリと公式キットを導入します。システム全体や予備戦環境の設定は変更しません。配布物と公式ランタイムはこのGit保存に含まれ、再ダウンロードは不要です。

自動検出されない場合は `./setup.ps1 -Python <python.exeのパス> -Uv <uv.exeのパス>` で指定できます。uvがなければ通常のvenv/pipを使います。uvだけある場合はPython 3.12を `.tools/python/` へ導入します。`-Offline` はuvと保存済みキャッシュがある場合に使えます。今回、新しい作業コピーで環境の再構築と94ファイルの検証が成功しています。

独自コードを動かすときのPythonは `.venv/Scripts/python.exe` です。日本語を扱う際は `$env:PYTHONUTF8='1'`、図を描く際は `$env:MPLCONFIGDIR="$PWD/local-output/matplotlib"` を指定します。`attack.ps1` はこれらを自動設定します。

## ファイルの場所

| 内容 | 場所 |
|---|---|
| 共通の模擬問題生成・保存・手法比較 | [../mock_test/](../mock_test/README.md) |
| TS本人の攻撃・学習済みモデル・過去の評価 | [attack_TS/](attack_TS/README.md) |
| 作業者Mの独立攻撃・予測・評価結果 | [Attack_M/](Attack_M/README.md) |
| 配布案内の本文 | [source_email_20260927.txt](source_email_20260927.txt) |
| 取得元・固定コミット・ファイルのSHA-256 | [source_manifest.json](source_manifest.json) |
| ダウンロード原本（攻撃ZIP・PDF・外部ハッシュ一覧・ソースZIP） | `downloads/` |
| 本戦のC 23件・AIA challenge 23件・対象一覧・内部ハッシュ一覧 | `targets/` |
| 本戦攻撃版の公式キット原本 | `official-kit/` |
| 自チームの本戦B・B_self・A_bg・schema・utility_ref・token | [../anonymization/team10/participant_data/](../anonymization/team10/participant_data/) |
| 検証と参照攻撃の入口 | [attack.ps1](attack.ps1) / [attack_workspace.py](attack_workspace.py) |
| 環境確認結果 | `local-output/environment_check.json` |
| 参照攻撃のCSVと動作確認結果 | `local-output/reference-baseline/` |
| GitHub保存用に固定した同じ参照出力 | [reference_baseline/](reference_baseline/) |
| 現在のルールブック | [../rulebook/PWS_Cup2026_ルールブック_main-process-20260912.pdf](../rulebook/PWS_Cup2026_ルールブック_main-process-20260912.pdf) |

配布ダウンロード6件、攻撃対象49件、公式キットの実行コードとライセンス、固定した参照出力を非公開Gitに保存します。公式ソースZIPには公開練習用のサンプルトークンが含まれますが、入れ子のZIPも含めて実team10トークンの混入がないことを確認しました。仮想環境・キャッシュ・通常の実験出力はGit管理対象外です。実トークンは既存の配置を参照し、今回新たにコピー・アップロードしたり、内容・ハッシュをログやマニフェストに残したりしません。

## 実データの参照と検証

本戦のBは1,049行、A_bgは5,143行です。`targets/mia_columns.json` の23コホートは **0〜17、22、25、26、27、28**。自己10を除く22対象に回答します。

| 出力 | 本戦team10の形式 |
|---|---|
| `F_mia.csv` | 1,049行 × 23列（`record_id` + 他22対象）。行IDは本戦Bの全ID。確信度は有限な0〜1 |
| `F_aia.csv` | 7,172行（326 × 22）× 3列。`target_k,challenge_row_id,time_hat`。各対象・行の組を1回ずつ回答 |
| 将来の提出ZIP | 直下に `F_mia.csv`、`F_aia.csv`、`token.txt` の3ファイル |

公式参照CLIは練習用の `B_practice.csv` を前提とし、公式検証CLIも `B.csv` から行IDを読みません。このため専用の入口から、**公式の攻撃関数・検証関数に実BのIDとteam_id=10を明示して渡します**。行IDの整数性、有限値、対象とchallengeの過不足も検査します。公式ソースや配布CSVは変更しません。

公式ソースZIPに含まれる `participant_data/` は公式の**練習用データ**です。元PCでは `official-kit/participant_data/` にも展開されていますが、この展開先はGit保存せず、本戦入力にも使いません。参照攻撃は実際の本戦BとA_bgを直接読みます。B_selfとutility_refは自己採点用の原本として保持し、他チームへの参照攻撃の入力には使いません。

`baseline` は環境の動作確認と今後の比較用です。手法を最適化した提出候補ではなく、公式攻撃力の得点も算出していません。ライブトークン入りZIPは自動生成しません。`validate` はローカルの形式・内容・自チームトークンとの一致を確認するもので、CodaBench上の登録状態や最終得点は確認しません。

## 配布元・整合性

- [公式release main-attack-20260928](https://github.com/pwscup/pwscup2026-kit/releases/tag/main-attack-20260928)
- キットの固定コミット：`a490490e11a0ec460a5df9fdefb17c72ad459c1e`
- 攻撃ZIP：`bdab1804efd39e6e227220ede168d2031fe1275b1b46655ef7891116a3160dc3`
- ZIP内の48ファイルは内部・外部の `SHA256SUMS.txt` と一致。ハッシュ一覧自体を含めて49ファイルを展開しました。
- PDFは既存本戦版と同一（SHA-256 `9290c6e98b85214ca691bb90ff961196642c72d85348810f8df36741489885f2`）。releaseの説明でもルール変更なし、検証器更新と明記されています。
- team10の既存配布原本6件は、9月16日の保存マニフェストと一致しました。
- 配布 `C_10.csv` は事務局メールのSHA-256 `7bcc96898b3d6bf4af8b7966d481fd0136dc64bbf3b68d0a5dd751a0b859c7db` と一致し、事務局掲載の提出IDは943347です。**GitHubの既存保存コミット51efc4fにある最終モデル[M1_v5/C.csv](../anonymization/team10/M/M1_v5/C.csv)とのバイト一致も確認しました。** 初回準備時点の「最終提出原本との照合は未確認」は解消済みです。攻撃対象からは自己10を除外します。

## 提出時の要点

採点まで通ると、リーダーボード掲載の有無に関係なく1回消費します。Failedとなる形式・値域違反は回数に算入しないと案内されています。順位に使うのは終了時点でリーダーボードに掲載された1件です。

事務局はCodaBenchのRunning/Scoringが完了しない不具合を案内しています。該当提出のCancelに伴う回数の差し戻しは事務局へ相談する扱いです。締切は10月11日の**午前9時**です。詳細は保存した案内本文と[公式CodaBench](https://www.codabench.org/competitions/17698)を参照してください。
