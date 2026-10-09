# Rv7：共分散を考慮するMIA

背景共分散を縮小推定した距離を用い、7連続属性と5連続QIの2通りで順位を補正します。時刻をMIAの追加信号に使わず、AIAはRv2と同じです。

**公式・新採点の得点は未測定です。Rv2を上回ることは未確認です。** 杉山採点で選択・調整していません。数値パラメーターと共通実装は本番予測前に固定しています。

- [設計・検証・共通実装](../rv6_series/README.md)
- [新採点向けAPIと比較時の注意](../rv6_series/EVALUATOR_HANDOFF.md)
- [MIA提出CSV](submission/F_mia.csv) / [AIA提出CSV](submission/F_aia.csv)
- [提出検証記録](verification.json) / [本番入力での変更数](production_diagnostics.json)
- PC提出ZIP：`submission/team10_Rv7_attack.zip`。認証情報を含むためGit除外です。

GitHubの提出データ保存先：
https://github.com/tsugiyama0216/pwscup2026-kit-private/tree/main/main_round/anonymization/team10/attack/attackR/Rv7/submission

再生成はリポジトリルートから次を実行します。出力先は新しいフォルダーにしてください。

```powershell
./main_round/anonymization/team10/attack/attackR/Rv7/run.ps1 -OutputDirectory ./main_round/anonymization/team10/attack/attackR/Rv7/reproduced
```

既存の提出CSV/ZIPは上書きしません。CodaBenchへの提出は行っていません。
