# TS（本人）の本戦攻撃

**別PCで取得した方へ：** v88は `prelim_transfer.methods` の元コードが不足しています。[別PC向け引き継ぎ](../../../CODEX_HANDOFF.md#v88_cross_pc_handoff)に、必要な関数・元PCでの探索先の候補・復旧後の確認手順を記載しています。`published_v88.check` の成功だけでは実行準備完了になりません。

ここは **TS（リポジトリ所有者本人）が作成した本戦攻撃** の保存先です。2026-09-30の本人の指定により、Mの攻撃と区別して `attack_TS/` にまとめました。作業者Mの独立攻撃は [../M/](../M/README.md) にあります。

共通の模擬問題生成・採点は [../../mock_test/](../../mock_test/README.md) へ分離しました。同じ保存問題へTSやMの攻撃を適用できます。TSの学習済みモデル・選択記録・過去の評価結果はこのフォルダーに保持します。

```text
main_round/attack/
├── attack_TS/                    # TS本人の攻撃
│   ├── attack_research/          # 各候補の実装とv88の再現入口
│   ├── local-output/attack-research/ # 保存済み学習モデル・選択記録
│   └── layout_migration.json     # 移動前後の対応・ハッシュ照合
├── M/                            # 作業者Mの攻撃と予測・評価結果
└── targets/・official-kit/など    # 両者が参照する共通の配布物・実行環境
```

| 内容 | 場所 |
|---|---|
| v88の保存済み攻撃と再現手順 | [attack_research/published_v88/README.md](attack_research/published_v88/README.md) |
| 攻撃実装 | [attack_research/](attack_research/) |
| 共通の模擬問題の生成・評価手順 | [../../mock_test/README.md](../../mock_test/README.md) |
| 保存済み学習モデル・候補選択の記録 | [local-output/attack-research/](local-output/attack-research/) |
| 配布物・公式検証・共通環境のセットアップ | [../README.md](../README.md) |

リポジトリ直下からの作業開始例です。共通の専用Python環境がまだなければ、先に `main_round/attack/setup.ps1` で用意します。

```powershell
Set-Location main_round/attack/attack_TS
..\.venv\Scripts\python.exe -m attack_research.published_v88.check
# v88の不足依存を復旧した後、共通評価器へ渡す場合
..\..\mock_test\evaluate.ps1 -Split dev -Plugin .\attack_research\published_v88\plugin.py -NullRepetitions 0
```

既存のモデル・選択記録・Mの予測CSV・配布原本は内容を保持しています。実行時の参照先だけを新配置へ合わせ、移動前のv88マニフェストは `attack_research/published_v88/manifest_before_attack_TS.json` に残しました。過去の生成・検証JSON内のパスは作成当時の記録として保持し、現在の配置は `layout_migration.json` を参照します。今回の整理でモデルの再学習や予測の再生成はしていません。

最初のTS整理時の記録では、原本94件・v88構成106件の照合、学習済みモデル11ファイルの読み込み、Mの既存テスト11件とTSの生成器テスト2件を確認しました。その時点では本戦専用環境は未構築でした。また、取得時点のリポジトリにはv88が必要とする `prelim_transfer.methods` が元から含まれておらず、v88全体の起動にはこの依存コードの復元も必要です。今回の共通ツールへの分離でも、この不足は未解決です。

その後、同日にこのPCの本戦専用環境を構築し、原本94件の照合と公式指定29ライブラリの版を確認しました。現在はその環境を共通模擬テストにも使用します。v88全体の実行成功を示す確認ではありません。

## Mのv2との共有環境の互換性

Mのv2をmainへ統合する際、Mが固定した共有コードと `attack/local_eval/` の3ファイルを元のバイトで保持しました。この `local_eval` はM v1/v2の互換用であり、共通評価器の正本は `main_round/mock_test/local_eval/` です。TSの実行入口は従来どおりこのフォルダーで使います。`_workspace_compat.py` が共通ツールの初期化処理を呼び、必要なパスとTS用の保存先属性を設定します。Mの固定コード・ハッシュ検査を変更せずに両立させます。

統合前のv88マニフェストは `attack_research/published_v88/manifest_before_M_v2_main.json` に保存し、有効な `manifest.json` には互換処理と共有コードの正確なハッシュを記録しています。過去の `layout_migration.json` と生成・検証記録は当時の履歴として保持します。これは配置の互換対応であり、欠落した `prelim_transfer.methods` を復元するものではありません。
