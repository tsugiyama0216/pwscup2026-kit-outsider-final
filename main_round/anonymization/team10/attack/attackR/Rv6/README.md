# Rv6：複数近傍の局所密度MIA

7属性と時刻を加えた8属性の距離比を1・3・10近傍で集約し、Rv2の順位と固定比率で混合します。AIAはRv2と同じです。

**公式・新採点の得点は未測定です。Rv2を上回ることは未確認です。** 杉山採点で選択・調整していません。数値パラメーターと共通実装は本番予測前に固定しています。

- [設計・検証・共通実装](../rv6_series/README.md)
- [新採点向けAPIと比較時の注意](../rv6_series/EVALUATOR_HANDOFF.md)
- [MIA提出CSV](submission/F_mia.csv) / [AIA提出CSV](submission/F_aia.csv)
- [提出検証記録](verification.json) / [本番入力での変更数](production_diagnostics.json)
- PC提出ZIP：`submission/team10_Rv6_attack.zip`。認証情報を含むためGit除外です。

GitHubの提出データ保存先：
https://github.com/tsugiyama0216/pwscup2026-kit-private/tree/main/main_round/anonymization/team10/attack/attackR/Rv6/submission

再生成はリポジトリルートから次を実行します。出力先は新しいフォルダーにしてください。

```powershell
./main_round/anonymization/team10/attack/attackR/Rv6/run.ps1 -OutputDirectory ./main_round/anonymization/team10/attack/attackR/Rv6/reproduced
```

既存の提出CSV/ZIPは上書きしません。CodaBenchへの提出は行っていません。
