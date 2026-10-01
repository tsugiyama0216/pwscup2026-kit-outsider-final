# 作業者Mの本戦攻撃

2026-10-01にフォルダー名を `M` から **`Attack_M`** へ変更しました。正本と保存結果はこのフォルダーです。実行は `./main_round/attack/Attack_M/run.ps1 M-v2-test` を使用します。旧名 `M` は固定コードのimport・ハッシュ検証を保つために自動生成するGit管理外の互換コピーで、編集しません。出力先は `Attack_M/local-output/` と共有します。初回の環境構築では親の `setup.ps1` が互換用の配置も準備します。v1/v2内の固定済み文書・JSONに残る旧パスは当時の記録として保持しています。

このフォルダーは、作業者 **M（ユーザー本人）** が作成した本戦攻撃をまとめる場所です。`M` は作業者、`v1`・`v2` はその作業者の版を表します。最新は **Mのv2** です。

v2の保存先は `main` です。固定コード・設定・予測CSVを保持し、TSの配置と両立するよう統合しました。[統合検証記録](main_integration_20260930.json)を参照してください。先行保存の `codex/m-v2` ブランチも履歴として残しています。

2026年9月30日、[v2](v2/README.md)の実装・固定後の模擬検証・22相手分の予測作成を完了しました。県による照合候補の不足に対応するAIAと、相手間の共通傾向を差し引くMIAです。公式形式・2回生成のバイト一致を確認済み。v1を保持し、公式提出はしていません。

複数の模擬問題で手法を比較する入口は [../../mock_test/](../../mock_test/README.md) です。共通ツールから `Attack_M/v1/model.py` をプラグインとして読み込み、既定の採用方式 `full_reference_aia` を同じ保存問題に適用できます。`main_round/mock_test/` での例：

```powershell
.\generate.ps1 -Split dev -Seed 20260927 -Count 3
.\evaluate.ps1 -CaseDirectory .\local-output\shadow-cases\dev-20260927-n3 -Plugin ..\attack\Attack_M\v1\model.py
```

この共通ツールの新しい問題・比較結果は `main_round/mock_test/local-output/` に保存します。以下の `Attack_M/local-output/v1/` は保存済みの本番予測・診断・評価記録で、内容を保持します。既存の `M-v1-evaluate` はMの固定条件を再実行する専用手順として残します。共通評価では自己163人マスクの実戦用処理を含まず、異なる抽出seedの得点を過去の評価と直接比較しません。

この例は単一標的APIを持つv1向けです。v2の `predict_all()` は複数標的を同時に受け取り、相手間の中央値を使うため、`Attack_M/v2/model.py` をそのまま共通ツールの `-Plugin` へ渡すことはできません。v2の固定済み多標的検証は [v2の手順](v2/README.md) を使用します。

M v1/v2の固定依存を保つため、`main_round/attack/local_eval/` の3ファイルと共有実行コードは元のバイトで保持します。この互換スナップショットへ新しい共通評価機能は追加せず、共通ツールは `main_round/mock_test/` で管理します。過去の固定ハッシュや評価記録は変更しません。

```text
main_round/attack/Attack_M/
├── README.md
├── v1/                        # Mのv1：コード・設定・採用理由
├── v2/                        # Mのv2：コード・固定設定・検証結果
├── local-output/
│   ├── v1/
│   │   ├── predictions/       # F_mia.csv・F_aia.csv・検証記録
│   │   ├── diagnosis/         # 対象別診断
│   │   ├── dev/・dev-final/   # 開発中の比較
│   │   └── holdout/           # 方式固定後の確認
│   ├── v1-planning-20260930/  # 初期作戦の監査記録
│   └── v2/                  # 調整・追加ストレス検査・最終確認・予測CSV
└── strategy/                  # Mの作戦と根拠
```

| 内容 | 場所 |
|---|---|
| Mのv2の説明・実行手順 | [v2/README.md](v2/README.md) |
| v2 MIA予測CSV | [F_mia.csv](local-output/v2/predictions/F_mia.csv) |
| v2 AIA予測CSV | [F_aia.csv](local-output/v2/predictions/F_aia.csv) |
| Mのv1の説明・実行手順 | [v1/README.md](v1/README.md) |
| v1 MIA予測CSV | [F_mia.csv](local-output/v1/predictions/F_mia.csv) |
| v1 AIA予測CSV | [F_aia.csv](local-output/v1/predictions/F_aia.csv) |
| 作戦の記録 | [V1_STRATEGY.md](strategy/V1_STRATEGY.md) |
| 旧配置との対応と検証 | [layout_migration.json](layout_migration.json) |

以後、Mの新しい版は `Attack_M/v3/`、その出力は `Attack_M/local-output/v3/` のように追加します。共有の配布原本・公式キット・Python環境は従来どおり `main_round/attack/` を参照します。

ルートから `./main_round/attack/Attack_M/run.ps1 M-v2-test` でv2を検証できます。v1は `M-v1-test`、旧 `v1-*` コマンドもMのv1へ接続します。

2026年9月30日の整理では、攻撃手法・採用モデル・保存CSV・生成時のJSON記録を保持しました。過去のJSONに残る旧パス・ハッシュは生成時点の履歴です。配置変更は上記の移行記録にまとめています。進行中の研究が参照する旧パスのファイルは、このPCだけの互換コピーとして残し、GitHubの正本はこのM配下に統一します。
