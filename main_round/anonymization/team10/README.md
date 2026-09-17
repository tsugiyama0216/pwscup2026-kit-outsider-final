# 本戦・加工フェーズ：team10 配布データ

2026-09-16にユーザー提供の配布データをこのPCへ配置しました。**本戦用の実データ**です。チーム名はアウトサイダー / outsider、コホート番号は **10**。予備戦のteam24および予備戦の練習データとは別に管理します。

配布元フォルダー名：`PWSCup2026_final_team10_アウトサイダー_outsider`（ユーザーのDownloads配下）。元ファイルはそのまま残し、内容を変更せずコピーしました。

配布README記載の提出締切：**2026-09-25 09:00（日本時間）**。

## ファイル

| ファイル | 用途 | 保存先 |
|---|---|---|
| [B.csv](participant_data/B.csv) | 自コホートの加工対象データ（配布形式14列） | PC・非公開GitHub |
| [B_self.csv](participant_data/B_self.csv) | 真の希少ラベル `is_rare` を含む自己採点用データ | PC・非公開GitHub |
| [A_bg.csv](participant_data/A_bg.csv) | 本戦の背景データ | PC・非公開GitHub |
| [schema.json](participant_data/schema.json) | 列定義・希少検出の参照 | PC・非公開GitHub |
| [utility_ref.json](participant_data/utility_ref.json) | 自己採点の参照値 | PC・非公開GitHub |
| [README.txt](participant_data/README.txt) | 事務局の配布案内原本 | PC・非公開GitHub |
| `participant_data/token.txt` | 提出用の秘密トークン | PC・非公開GitHub（2026-09-17に明示許可） |

[data_manifest.json](data_manifest.json) に原本6ファイルのサイズとSHA-256を記録しています。トークンの内容とハッシュは記載していません。

## 作業者ごとのモデル格納先

4人のモデルは、このフォルダー直下の作業者別フォルダーで管理します。

| 作業者 | 格納先 | モデルフォルダーの命名 |
|---|---|---|
| S | [S/](S/) | `S1/`、`S2/`、`S3/` … |
| R | [R/](R/) | `R1/`、`R2/`、`R3/` … |
| M | [M/](M/) | `M1/`、`M2/`、`M3/` … |
| L | [L/](L/) | `L1/`、`L2/`、`L3/` … |

番号は作業者ごとに1から始めます。各モデルの実装・設定・評価結果をモデルごとのフォルダーにまとめ、配布原本は共通の `participant_data/` を参照します。2026-09-17に [S/S1](S/S1/README.md) を作成しました。予備戦提出v5の手法による本戦C.csvと提出ZIPは形式検証済みですが、本戦の相関指標が0点となり自己採点U=0です。CodaBenchには未提出です。

## 保存・利用について

- 2026-09-16のユーザーの明示的依頼に基づき、配布原本6ファイルを既存の非公開リポジトリ `tsugiyama0216/pwscup2026-kit-private` へ保存します。リポジトリを公開せず、配布元READMEの取扱条件に従ってください。
- 2026-09-17のユーザーの明示的指定により、team10の `participant_data/token.txt` と [`模擬提出データ/`](模擬提出データ/) 内の `token.txt` も、同じ非公開リポジトリに保存します。追加の明示的な許可により、[S1のトークン入り提出ZIP](S/S1/submission/team10_S1_defense.zip)もGit管理します。その他のトークン・ZIPとローカル実行出力は引き続き除外します。トークンの内容は文書やログに記載しません。
- S1で本戦公式キットmain-process-20260912の導入・動作検証、C.csv生成、提出ZIP作成・形式検証・自己採点を実施済みです。実行環境と評価記録は [S/S1](S/S1/README.md) に保存しています。公式提出は行っていません。
- 本戦用キットを導入するときは、このフォルダーのデータを明示的に参照します。`prelim/workspace/participant_data/` の練習用データや予備戦team24の実データを混ぜないでください。
- 提出Cの列定義は本戦の `schema.json` にある `submitted_c_columns`（`record_id` を除いた13列）を参照します。

[本戦の全体案内](../../README.md)・[本戦ルール](../../rulebook/README.md)
