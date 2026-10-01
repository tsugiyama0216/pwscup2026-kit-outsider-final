# TS本人の攻撃・第88候補の再現一式

2026-09-30に不足依存を復旧しました。ローカルに残っていた元の `prelim_transfer` アダプター2ファイルを `Attak_T/prelim_transfer/` へバイト一致で回収し、予備戦の依存ソース16件をマニフェストに追加しました。Git管理ソースに復旧対象だけを追加した独立コピーで、130ファイル・原本94件の照合、v88初期化、固定25条件benchmarkの再現に成功しました。[復旧検証記録](dependency_recovery_20260930.json)を参照してください。

別PCで利用する担当者・エージェントは、[復旧内容と確認手順の引き継ぎ](../../../../../CODEX_HANDOFF.md#v88_cross_pc_handoff)を先に読んでください。`check` はハッシュ照合、`check --runtime` は照合に加えて正規のアダプター・予備戦コードの読み込み元と必要な関数を確認し、v88を初期化します。予測と採点の再現は `benchmark` で確認します。今回の独立コピーの検証は同じPC上で既存の固定Pythonを隔離モードで使い、そのコピーの公式ソースを明示して実施しました。別の実機PCでの実行は未確認です。

ここには保存済みv88の実行入口・関連コード・学習済みモデルがあります。実装は `attack_research/v88/predict.py` から過去版の部品を呼び出し、回収した `prelim_transfer` が `prelim/workspace/` の数学的処理へ接続します。アルゴリズム・学習済みモデル・候補選択は変更していません。第89〜99候補など不採用の実験、学習用の巨大な中間配列、予測済みの本戦回答CSVも含めていません。

別PCでこの非公開リポジトリを取得したら、Windows PowerShellで`main_round/attack/Attak_T/`へ移動して次を実行します。共通Python環境が構築済みならセットアップは不要です。

```powershell
..\setup.ps1
..\.venv\Scripts\python.exe -m attack_research.published_v88.check --runtime
..\.venv\Scripts\python.exe -m attack_research.published_v88.benchmark
```

`check --runtime` は同梱コード・学習済みモデルと配布原本94件を照合し、依存元と初期化を確認します。`benchmark` は固定した5抽出×5加工方式を新たに生成し、MIA平均 `0.7307692307692308`、AIA平均 `0.8473837794327597` を許容差 `1e-10` で照合し、両方1.0が `8/25` 条件になることを確認します。今回の独立コピーの検証でこの条件を満たしました。AIAは未知の公式補正φを引かないローカル代理スコアであり、本戦の公式得点ではありません。

別の模擬条件を比較する場合は、公開入力だけを受け取る `plugin.py` を共通評価器に渡せます。作業場所は引き続き `main_round/attack/Attak_T/` です。

```powershell
..\..\mock_test\evaluate.ps1 -Split dev -Plugin .\attack_research\published_v88\plugin.py -NullRepetitions 0
```

このコマンドの既定3抽出・主15条件は上記の固定25条件とは異なります。単一抽出の主5条件や別seedの結果とも直接比較しません。共通ツールの `-CaseDirectory` で同じ保存スイートを指定すると、他の攻撃と同条件で比較できます。生成・保存・採点は [共通模擬テスト](../../../../mock_test/README.md)、公開/正解ファイルの分離とAPIは [評価器の詳細](../../../../mock_test/local_eval/README.md) を参照してください。本戦22対象の回答CSVを生成・提出するコマンドではありません。

2026-09-30にTS本人の攻撃として `Attak_T/` へ移動しました。学習済みモデル・選択JSONは同じバイトを保持し、実行時のパスだけを更新しています。移動前の固定マニフェストは [manifest_before_attack_TS.json](manifest_before_attack_TS.json)、移動後のコードと配置の検査は [manifest.json](manifest.json)、移動前後の照合は [layout_migration.json](../../layout_migration.json) を参照します。

同日の追加整理で `local_eval/` と生成・評価スクリプトを `main_round/mock_test/` へ分離しました。分離前のマニフェストは [manifest_before_mock_test.json](manifest_before_mock_test.json) に保持します。今回の復旧後の `manifest.json` はリポジトリ直下を基準に、本戦の実行コードと共通ツール、予備戦の依存コードを照合します。以前のハッシュ照合・モデル11件の読み込み成功だけでは起動を確認できなかった経緯も、復旧記録に残しています。既存モデル・選択JSON・過去の検証記録の内容は維持します。
