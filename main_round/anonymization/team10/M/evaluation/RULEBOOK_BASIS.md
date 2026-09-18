# 耐性評価のルールブック根拠

参照原本は [`../../../../rulebook/PWS_Cup2026_ルールブック_main-process-20260912.pdf`](../../../../rulebook/PWS_Cup2026_ルールブック_main-process-20260912.pdf) です。

## 匿名性の集約（§6.2、p.19）

匿名性はMIAとAIAの悪い方で決まります。

```text
Anon(i) = 1 - max(a_MIA(i), max_j a_AIA(i;j))
```

ローカル評価でもMIA・AIAの最大リスクを使います。ただし他チームの攻撃がないため、複数のローカル攻撃の最大を取るストレステストに置き換えています。

## MIA（§6.2.1、pp.19–20）

- ROCを作り、FPR=0.01で線形補間したTPRを読む。
- 全候補と希少候補の大きい方を攻撃成功度とする。
- 正解membershipと希少フラグを対にして20,000回並べ替え、集約値の95パーセンタイルを偶然水準とする。
- `beta = max(beta_raw - beta_chance + 0.01, 0.01)`。
- `a_MIA = clip((beta/0.01 - 1)/(K - 1), 0, 1)`。固定キットのKは12。

ローカル候補集合は公式の25–27人member構成を再現できないため、K=12正規化値は追跡用に残しますが、モデル比較には飽和前のbetaを使います。

## AIA（§6.2.2、pp.20–21）

`time_hat` が真値の±0.5年以内なら正解です。標的正解率を `p_m`、対照正解率を `p_c` とし、次で攻撃成功度を測ります。

```text
R = clip((p_m - p_c) / (1 - p_c), 0, 1)
```

希少側にも同じ計算を行い、固定キットの `z=1.64` の標準誤差デッドゾーンを適用して全体との悪い方を取ります。本戦ではさらにコホート別 `phi_i` を控除しますが、team10の値が保存物にないため、ローカル値は控除前です。

## 提出判断用の値

公式総合順位は `min(U, Anon)` です（§5.6、p.12）。ローカル比較では次を表示します。

```text
diagnostic_attack_risk = max(MIA beta, AIA R before phi)
diagnostic_resistance = 1 - diagnostic_attack_risk
screening_floor = min(U, diagnostic_resistance)
```

これはモデル間の比較を素早く行うためのローカル指標で、公式protection、Anon、総合得点の予測ではありません。
