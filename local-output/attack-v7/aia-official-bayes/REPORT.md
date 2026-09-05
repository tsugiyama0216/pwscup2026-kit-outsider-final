# AIA official-feedback Bayesian/robust re-evaluation

## 結論

正常形式の明確な改善候補は `complement_unanimous_active`。共通0点target 6, 7, 8, 9, 17, 18, 19で、
v4・independent・routedの三方式が**全会一致でactive**にした行だけを`11.0`へ退避し、
それ以外を`9.5`にする。現在のrouted完全補集合をすべて含み、541行を追加する。

探索側37 blockでは完全補集合比の対象別power差が平均
`+0.033726`、勝率`64.9%`。
未使用の検証側26 blockでは平均`+0.048263`、
勝率`76.9%`、p10差`-0.043660`だった。
score_1への方向的換算は約`+0.016892`。これはprivate score予測ではない。

## Production変換仕様

| target | active rows | exact complementから追加 | unanimous-activeで退避 |
|---:|---:|---:|---:|
| 6 | 217 | 54 | 109 |
| 7 | 253 | 90 | 73 |
| 8 | 211 | 48 | 115 |
| 9 | 274 | 111 | 52 |
| 17 | 223 | 60 | 103 |
| 18 | 236 | 73 | 90 |
| 19 | 268 | 105 | 58 |

判定は各公式提出ファイルの`time_hat != 11.0`をactiveとする。3つすべてactiveなら11.0、
それ以外なら9.5。正点13 targetとF_miaは変更しない。

## 比較結果

- exact-zero検証: baseline `0.139850` → candidate `0.188114`。
- near-zero検証35 blockでも平均差 `+0.044227`、勝率 `74.3%`。
- active数を32〜144へ絞る案の検証最良は `ridge_100_global_k144` だが、平均差は
  `+0.000415`、勝率`46.2%`で再現性がなく棄却。
- `000`だけ、active多数決の補集合、release距離ridge/Bayesian tailも完全補集合を安定して上回らなかった。

## 公式制約と限界

三つのproduction maskは提出済みroutedを含め2,282/2,282行で再構成一致した。routedの
公式集計は、7 target合計でactive側のcontrol成功がmember成功より59〜68件多い。
一方、公式集計だけでは三方式全会一致集合の符号は識別できず、136 hits/arm/targetを固定した
MILPでもcandidate raw advantageは`-418`〜
`486`。したがって最悪値保証ではなく、独立seedで再現した
条件付きshadowの順位を採用根拠とする。

## QA

- research mask: 2,282行、複合key重複なし、全challenge keyと一致
- time_hat: 9.5 / 11.0のみ
- production candidate・ZIP・token・提出: 変更なし
