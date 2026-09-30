# 作業者Mの本戦攻撃

このフォルダーは、作業者 **M（ユーザー本人）** が作成した本戦攻撃をまとめる場所です。`M` は作業者、`v1`・`v2` はその作業者の版を表します。最新は **Mのv2** です。

v2の再現用保存先は `codex/m-v2` ブランチです。main側の別配置への変更と固定依存を混在させないため、作成時点の環境をこのブランチに保持しています。

2026年9月30日、[v2](v2/README.md)の実装・固定後の模擬検証・22相手分の予測作成を完了しました。県による照合候補の不足に対応するAIAと、相手間の共通傾向を差し引くMIAです。公式形式・2回生成のバイト一致を確認済み。v1を保持し、公式提出はしていません。

```text
main_round/attack/M/
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

以後、Mの新しい版は `M/v3/`、その出力は `M/local-output/v3/` のように追加します。共有の配布原本・公式キット・Python環境は従来どおり `main_round/attack/` を参照します。

ルートから `./main_round/attack/attack.ps1 M-v2-test` でv2を検証できます。v1は `M-v1-test`、旧 `v1-*` コマンドもMのv1へ接続します。

2026年9月30日の整理では、攻撃手法・採用モデル・保存CSV・生成時のJSON記録を保持しました。過去のJSONに残る旧パス・ハッシュは生成時点の履歴です。配置変更は上記の移行記録にまとめています。進行中の研究が参照する旧パスのファイルは、このPCだけの互換コピーとして残し、GitHubの正本はこのM配下に統一します。
