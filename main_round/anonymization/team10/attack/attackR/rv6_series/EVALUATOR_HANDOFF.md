# 未唯奈さんの新採点システムへの引き継ぎ

## 受け取るもの

Rv6〜Rv11の各フォルダーに本戦公開データへの予測CSV、共通フォルダーに方式・固定パラメーター・再現用コードがあります。既存Rv2の公式フィードバックは出典として添付しています。新6版の得点は未測定です。

本戦の提出CSVに模擬正解を組み合わせないでください。異なるデータ世界では、その世界の公開入力からRv2の基準予測と新候補の予測をそれぞれ再生成する必要があります。本戦の正解CSVは保持していません。

## 一括公開入力API

`rv6_series/model.py` の `predict_all` はファイル、採点器、正解を読みません。`runtime.py` は本番専用の入出力です。

```python
outputs, diagnostics = predict_all(
    candidates, background, targets, challenges,
    baseline_mia, baseline_aia, known_negative, settings,
)
```

| 引数 | 内容 |
|---|---|
| candidates | 当該世界のMIA候補DataFrame。FULLの7属性、time、sex、prefectureが必要 |
| background | 当該世界の公開背景DataFrame。同じ属性が必要 |
| targets | `{int target_k: 公開CのDataFrame}`。最低2対象 |
| challenges | `{int target_k: 公開AIAのDataFrame}`。age,SBP,TG,HDL,ALT,sex,prefecture |
| baseline_mia | `{target_k: candidatesと同じ順序のRv2スコア配列}` |
| baseline_aia | `{target_k: challenges[k]と同じ順序のRv2時刻配列}` |
| known_negative | candidates順のbool配列。公開情報で導けない世界はすべてFalse |
| settings | 同梱 `protocol.json` を読み込んだ辞書 |

メンバー正解、AIA正解時刻、加工方式名、隠された希少ラベルは引数に渡しません。公開QI由来の背景分布推定と、真の公式希少ラベルは区別します。IDは位置合わせにのみ用い、特徴量には使いません。各参照集合は10行以上必要です。

戻り値は `{model_name: {'mia': {k: score_array}, 'aia': {k: time_array}}}` です。候補・問題の順序を保ちます。個別の `RvN/model.py` には同じ引数（settings以外）を持つ `predict` もあります。全版比較では一括APIのほうが計算を共有できます。

## 評価時の推奨記録

1. Rv2と6候補を同じ未使用データ世界で評価。方式開発に使った問題・人物・乱数系列との重複を記録。
2. 全体・希少層を分け、MIA TPR@FPR=0.01/AUC、AIA p_m/p_cと補正後指標を保存。FPR閾値決定の同点処理と母数も保存。
3. 公式の隠れた正解・対象別floorを再現できない部分は推定と明記。過去に模擬・公式で乖離したため、単一合成世界の順位を本戦の順位と解釈しない。
4. Rv6/7/8はMIAだけ、Rv9/10はAIAだけの比較ができる。Rv11は統合による相互作用を別に確認。
5. 候補ごとの対象別差、上位尾部の真陽性と偽陽性、希少層の少数サンプルによるばらつきを残す。対象を跨ぐ同一人物の相関を無視した信頼区間を避ける。
6. 選択用と最終確認用を分離。結果を見て改変する場合は新しい版名で保存し、Rv6〜Rv11の固定ファイルを上書きしない。

これらは新採点の仕様を勝手に固定するものではなく、比較の取り違えを防ぐための引き継ぎ事項です。
