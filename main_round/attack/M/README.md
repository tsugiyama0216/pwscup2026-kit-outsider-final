# 作業者Mの本戦攻撃

このフォルダーは、作業者 **M（ユーザー本人）** が作成した本戦攻撃をまとめる場所です。`M` は作業者、`v1` はその作業者の第1版を表します。名称は **Mのv1** です。

```text
main_round/attack/M/
├── README.md
├── v1/                        # Mのv1：コード・設定・採用理由
├── local-output/
│   ├── v1/
│   │   ├── predictions/       # F_mia.csv・F_aia.csv・検証記録
│   │   ├── diagnosis/         # 対象別診断
│   │   ├── dev/・dev-final/   # 開発中の比較
│   │   └── holdout/           # 方式固定後の確認
│   └── v1-planning-20260930/  # 初期作戦の監査記録
└── strategy/                  # Mの作戦と根拠
```

| 内容 | 場所 |
|---|---|
| Mのv1の説明・実行手順 | [v1/README.md](v1/README.md) |
| MIA予測CSV | [F_mia.csv](local-output/v1/predictions/F_mia.csv) |
| AIA予測CSV | [F_aia.csv](local-output/v1/predictions/F_aia.csv) |
| 作戦の記録 | [V1_STRATEGY.md](strategy/V1_STRATEGY.md) |
| 旧配置との対応と検証 | [layout_migration.json](layout_migration.json) |

以後、Mの新しい版は `M/v2/`、その出力は `M/local-output/v2/` のように追加します。共有の配布原本・公式キット・Python環境は従来どおり `main_round/attack/` を参照します。

ルートから `./main_round/attack/attack.ps1 M-v1-test` で検証できます。旧 `v1-*` コマンドもMのv1へ接続します。

2026年9月30日の整理では、攻撃手法・採用モデル・保存CSV・生成時のJSON記録を保持しました。過去のJSONに残る旧パス・ハッシュは生成時点の履歴です。配置変更は上記の移行記録にまとめています。進行中の研究が参照する旧パスのファイルは、このPCだけの互換コピーとして残し、GitHubの正本はこのM配下に統一します。
