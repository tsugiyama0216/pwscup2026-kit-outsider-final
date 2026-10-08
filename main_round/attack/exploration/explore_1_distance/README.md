# ① 距離によるMIAとAIA

このフォルダーは探索用です。採用済みの攻撃や公式提出物とは分けて保存します。

`plugin.py` は `predict_world` に渡された公開入力だけから予測します。予備戦の正解・他チームの回答・採点結果は読みません。パラメーターは採点前に固定し、正解を使った調整は行いません。

## 手法

- MIA: 候補者と合成表の最近傍距離に負号を付けます。小さい距離ほど所属を疑います。
- AIA: 同じ距離で最も近い合成行の `time` を回答します。最短距離が同じ行が複数ある場合は、その時間を値順で並べた平均です。
- 距離: `age, SBP, TG, HDL, ALT` を背景表の標準偏差（ddof=1）で割った後のRMS距離です。性別・県が一致する行に絞り、空なら性別、さらに空なら全体へ広げます。
- MIA出力は生スコアの順位を0〜1へ写した値です。較正済みの所属確率ではありません。同点は同点のままです。

根拠となった行の近さで両攻撃を説明できます。MIAにもQI7列だけを使うため、公式DOMIASの連続7列（BMI/FPGを含む）とは使用情報が異なります。結果差を距離と密度の違いだけに帰属しないでください。

固定条件は `config.json`、共通実装は `../explore_common/core.py`。一括実行と比較は親の `README.md` を参照してください。

## 出典

- [TAPAS ClosestDistanceMIA / ClosestDistanceAIA](https://tapas-privacy.readthedocs.io/en/latest/library-of-attacks.html)
- [大会公式AIA](https://github.com/pwscup/pwscup2026-kit/blob/main/starter/src/pwscup2026/attack/aia.py)
