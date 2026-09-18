# R3 耐性評価

- 有用性U: 0.866124
- MIA βリスク（偶然補正後・K正規化前）: 0.587465
- MIA K=12正規化リスク: 1.000000
- AIAリスク（φ控除前）: 0.065668
- 診断用耐性指数: 0.412535
- 提出比較用screening floor: 0.412535
- リスク帯: `high-local-signal`
- 判定: **耐性信号が強いため改善・再検証**

## 攻撃別

| 攻撃 | 主な値 |
|---|---:|
| MIA `official_domias_kde` β / normalized | 0.587465 / 1.000000 |
| MIA `team_v4_v9_multiview` β / normalized | 0.010000 / 0.000000 |
| AIA `official_qi_nearest` R | 0.065668 |
| AIA `team_multiview_linkage` R | 0.060569 |
| AIA `public_horizon_sanity` R | 0.047901 |

## 解釈上の制約

- Official cross-team MIA candidate sets and AIA target/control challenges are unavailable.
- The local proxy uses all B rows as members/targets and public A_bg rows as controls.
- The team10 official AIA phi_i baseline is unavailable, so the reported AIA risk is before phi subtraction.
- Multiple attacks are aggregated by maximum as a conservative development stress test, not by the official cross-team aggregation.
- K=12 normalization is reported for rulebook traceability, but the proxy candidate composition differs from the official 25-27-member pool and may saturate; model ranking therefore uses the chance-corrected beta scale before K normalization.
- Results compare models on this fixed local proxy and are not leaderboard predictions.

詳細JSON: [resistance_report.json](resistance_report.json)
