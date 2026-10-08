# ② DOMIASと近傍のtime分布

このフォルダーは探索用です。公開引数以外の大会データ・採点結果は攻撃コードに読み込みません。パラメーターは採点前に固定し、予備戦正解で調整しません。

## MIA

予備戦評価器が導入する固定版公式 `overlap_mia` をそのまま使います。連続7列 `age, BMI, SBP, TG, HDL, ALT, FPG` のKDE密度比 `p_C(x)/p_bg(x)` を計算し、公式の `ratio/(1+ratio)` で0〜1へ写します。背景表で標準化（ddof=0）、SciPy Gaussian KDE既定のScott帯域、特異共分散なら全候補0.5という公式動作も保持します。

## AIA

QIによるブロック・標準化は①および公式AIAと同じです。各challenge行の近傍10行を選び（10番目の距離が同点なら全部含める）、その `time` が最も多く入る幅1年の閉区間を求めます。最大数が同じ場合は最も早い区間を選び、その中の最小・最大timeの中点を回答します。近傍数に満たないブロックでは利用可能な全行を使います。

これはLocalNeighbourhoodAttackを連続時間と±0.5年の成功条件に合わせた変種です。TAPASのカテゴリ向け実装をそのまま再現したものではありません。距離重み・学習済みの混合・閾値探索は使いません。

①との違いにはMIA特徴列も含まれます。公式参照攻撃との比較ならMIAは同一で、AIAの集約方式の影響を分離できます。③との比較ならAIAは同一で、MIAの違いを比較できます。

固定条件は `config.json`、共通実装は `../explore_common/core.py`。実行と結果は親の `README.md` を参照してください。

## 出典

- [大会公式MIA](https://github.com/pwscup/pwscup2026-kit/blob/main/starter/src/pwscup2026/attack/mia.py)
- [TAPAS LocalNeighbourhoodAttack](https://tapas-privacy.readthedocs.io/en/latest/library-of-attacks.html)
