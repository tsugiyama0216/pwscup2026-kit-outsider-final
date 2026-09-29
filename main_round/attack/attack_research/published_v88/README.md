# 推奨攻撃・第88候補の再現一式

ここには現在の推奨攻撃だけを公開するための入口があります。実装は`attack_research/v88/predict.py`から過去版の必要な部品を継承するため、その**実行時依存コードのみ**と学習済みモデルを同梱しています。第89〜99候補など不採用の実験、学習用の巨大な中間配列、予測済みの本戦回答CSVは含めません。

別PCでこの非公開リポジトリを取得したら、Windows PowerShellで`main_round/attack/`へ移動して次を実行します。

```powershell
.\setup.ps1
.\.venv\Scripts\python.exe -m attack_research.published_v88.check
.\.venv\Scripts\python.exe -m attack_research.published_v88.benchmark
```

`check`は同梱コード・学習済みモデルと配布原本94件を照合します。`benchmark`は固定した5抽出×5加工方式を新たに生成し、MIA平均**0.730769**、AIA平均**0.847384**、両方1.0が**8/25条件**になることを確認します。AIAは未知の公式補正φを引かないローカル代理スコアであり、本戦の公式得点ではありません。

自分で別の模擬条件を比較する場合は、公開入力だけを受け取る`plugin.py`を評価器に渡せます。

```powershell
.\evaluate.ps1 -Split dev -Plugin .\attack_research\published_v88\plugin.py -NullRepetitions 0
```

このコマンドの既定3抽出は上記の固定25条件とは異なります。模擬問題の生成と公開/正解ファイルの分離は[評価器の説明](../../local_eval/README.md)を参照してください。本戦22対象の回答CSVを生成・提出するコマンドではありません。
