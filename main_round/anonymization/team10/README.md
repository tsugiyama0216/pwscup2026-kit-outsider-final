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
| `participant_data/token.txt` | 提出用の秘密トークン | PCのみ・Git管理対象外 |

[data_manifest.json](data_manifest.json) に原本6ファイルのサイズとSHA-256を記録しています。トークンの内容とハッシュは記載していません。

## 作業者ごとのモデル格納先

4人のモデルは、このフォルダー直下の作業者別フォルダーで管理します。

| 作業者 | 格納先 | モデルフォルダーの命名 |
|---|---|---|
| S | [S/](S/) | `S1/`、`S2/`、`S3/` … |
| R | [R/](R/) | `R1/`、`R2/`、`R3/` … |
| M | [M/](M/) | `M1/`、`M2/`、`M3/` … |
| L | [L/](L/) | `L1/`、`L2/`、`L3/` … |

番号は作業者ごとに1から始めます。例えばSの最初のモデルは `S/S1/` に格納します。各モデルの実装・設定・評価結果をモデルごとのフォルダーにまとめ、配布原本は共通の `participant_data/` を参照します。現在は格納先のみを用意しており、モデルは未登録です。

## 保存・利用について

- 2026-09-16のユーザーの明示的依頼に基づき、配布原本6ファイルを既存の非公開リポジトリ `tsugiyama0216/pwscup2026-kit-private` へ保存します。リポジトリを公開せず、配布元READMEの取扱条件に従ってください。
- `token.txt`、本戦フォルダー内のZIP、ローカル実行出力はGit管理から除外します。別PCで利用する際は、認証トークンを別途安全に配置してください。
- 本戦の採点キットの導入・動作検証、加工アルゴリズムの実装、C.csvの生成・提出はまだ行っていません。今回保存した配布データだけで自己採点環境の構築完了とは扱いません。
- 本戦用キットを導入するときは、このフォルダーのデータを明示的に参照します。`prelim/workspace/participant_data/` の練習用データや予備戦team24の実データを混ぜないでください。
- 提出Cの列定義は本戦の `schema.json` にある `submitted_c_columns`（`record_id` を除いた13列）を参照します。

[本戦の全体案内](../../README.md)・[本戦ルール](../../rulebook/README.md)
