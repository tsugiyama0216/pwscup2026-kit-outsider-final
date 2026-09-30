# TS本人の攻撃・第88候補の再現一式

2026-09-30の確認：保存ファイルのハッシュ照合と学習済みモデル11件の読み込みは成功しました。ただし取得時点のリポジトリに、依存する `prelim_transfer.methods` が含まれていないため、以下の `benchmark` や攻撃全体の実行にはこのコードの復元が必要です。この不足は `attack_TS/` への移動前から存在します。

別PCで復旧する担当者・エージェントは、[不足コードと確認手順の引き継ぎ](../../../../../CODEX_HANDOFF.md#v88_cross_pc_handoff)を先に読んでください。`check` はマニフェスト記載ファイルの整合性検査であり、欠落した依存コードの検出やv88全体の実行検査ではありません。

ここには保存済みv88の実行入口・関連コード・学習済みモデルがあります。実装は`attack_research/v88/predict.py`から過去版の部品を呼び出しますが、上記の `prelim_transfer` は未収録です。第89〜99候補など不採用の実験、学習用の巨大な中間配列、予測済みの本戦回答CSVも含めていません。

別PCでこの非公開リポジトリを取得したら、Windows PowerShellで`main_round/attack/attack_TS/`へ移動して次を実行します。共通Python環境が構築済みならセットアップは不要です。

```powershell
..\setup.ps1
..\.venv\Scripts\python.exe -m attack_research.published_v88.check
..\.venv\Scripts\python.exe -m attack_research.published_v88.benchmark
```

`check`は同梱コード・学習済みモデルと配布原本94件を照合します。`benchmark`は固定した5抽出×5加工方式を新たに生成し、MIA平均**0.730769**、AIA平均**0.847384**、両方1.0が**8/25条件**になることを確認します。AIAは未知の公式補正φを引かないローカル代理スコアであり、本戦の公式得点ではありません。

不足依存の復旧後、別の模擬条件を比較する場合は、公開入力だけを受け取る`plugin.py`を共通評価器に渡せます。作業場所は引き続き `main_round/attack/attack_TS/` です。

```powershell
..\..\mock_test\evaluate.ps1 -Split dev -Plugin .\attack_research\published_v88\plugin.py -NullRepetitions 0
```

このコマンドの既定3抽出・主15条件は上記の固定25条件とは異なります。単一抽出の主5条件や別seedの結果とも直接比較しません。共通ツールの `-CaseDirectory` で同じ保存スイートを指定すると、他の攻撃と同条件で比較できます。生成・保存・採点は [共通模擬テスト](../../../../mock_test/README.md)、公開/正解ファイルの分離とAPIは [評価器の詳細](../../../../mock_test/local_eval/README.md) を参照してください。本戦22対象の回答CSVを生成・提出するコマンドではありません。

2026-09-30にTS本人の攻撃として `attack_TS/` へ移動しました。学習済みモデル・選択JSONは同じバイトを保持し、実行時のパスだけを更新しています。移動前の固定マニフェストは [manifest_before_attack_TS.json](manifest_before_attack_TS.json)、移動後のコードと配置の検査は [manifest.json](manifest.json)、移動前後の照合は [layout_migration.json](../../layout_migration.json) を参照します。

同日の追加整理で `local_eval/` と生成・評価スクリプトを `main_round/mock_test/` へ分離しました。分離前のマニフェストは [manifest_before_mock_test.json](manifest_before_mock_test.json) に保持し、現在の `manifest.json` は共通ツールの新しい配置を参照します。既存モデル・選択JSON・過去の検証記録の内容は維持します。
