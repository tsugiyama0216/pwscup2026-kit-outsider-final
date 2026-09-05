# PWSCUP2026 最終攻撃ローカル再評価

## 結論

最終推奨は `attack-v6-final-horizon-edge.zip` である。公式フィードバックから選んだ
13個の正点blockと、共通7個の0点blockに対するrouted完全補集合というmember選別は維持し、
target 1/2以外のexact `time_hat=10.0`だけを`9.5`へ移した。

これは正常な提出形式を保ち、`0 < time_truth <= 10`、`tau_time=0.5`の下で行単位の
成功区間を`[9.5, 10]`から`[9.0, 10]`へ拡張する。paired shadowと公開時刻分布を
合わせた現行補集合版からの期待増分はscore_1換算で約`+0.0051〜+0.0068`、条件付きの
総合見込みは概ね`0.335〜0.352`である。ただしprivate真値がないため保証値ではない。

## 主要な根拠

- 公式3回すべてで0点だったtarget 6, 7, 8, 9, 17, 18, 19では、直近routedの
  control成功数がmemberより合計59〜68件多いことを公式集計から整数復元できる。
- 公式0条件を再現したvalidationではrouted完全補集合が平均`0.108333`で首位。
  inverse consensusは`0.105673`、v4補集合は`0.085874`、independent補集合は
  `0.061842`だった。
- inverse consensusは完全補集合から154 swapを行い、exchangeableモデルでは公式の
  逆方向信号を73.01%へ薄める。公式集計に整合するMILPでも優劣は識別不能なので採用しない。
- 正点11 targetの`10.0 -> 9.5` paired shadowはscore_1換算平均`+0.0024505`。
  target 5の1/24だけが対象別`-0.00155`で、残りは全ケース非悪化だった。
- 0点条件を再現できたtarget 19では時刻シフトが24/24非悪化、対象別平均`+0.00863`。
  公開B/A時刻分布モデルでは7 block合計でscore_1 `+0.0027〜+0.0043`を見込む。

## MIA

MIAはv4をbyte-identicalに維持する。Daybreakはallが`0.106219`へ小幅改善した一方、
rareが`0.333333`から`0.166667`へ半減し、AUCも`0.649339`から`0.635601`へ低下した。
guided tailはall/rareがv4と同じでAUC差も約`0.000006`に留まり、最後の一回で変更する
根拠にならない。

## 棄却案

- rare top-26: rare power平均`0.14365 -> 0.06589`、24ケース中17悪化。
- age近傍pair: 対象別平均`-0.0280`、24ケース中20悪化。
- event-q75移植: 平均は上がるが24ケース中8悪化、p10=`-0.0645`。
- inverse consensus: 条件付きvalidationで完全補集合を上回らず、公式制約でも頑健優位なし。
- fractional IDなどのvalidator/scorer差: ルールブックの「各challenge行へちょうど1行」に反するため不使用。

## 0.5目標との距離

実測winner合成の再構成値は約`0.308532`。0.5には共通7個の0点targetから合計約`3.83`、
平均約`0.547`の追加powerが必要である。公式制約と条件付きshadowは補集合への改善方向を
支持するが、この規模を支持してはいない。したがって0.5は目標として残るものの、ローカルで
説明可能な中心域は`0.335〜0.352`とする。

## 検証と成果物

- 公式package validator: PASS
- unit tests: 113/113 PASS
- `compileall`: PASS
- ZIP member: `F_mia.csv`, `F_aia.csv`, `token.txt`のみ
- ZIP SHA-256: `D0E93343805D2A303EA752FA4CF1E176DCFE80CD56F54488B3DD36C38ACBD1E6`
- `F_mia.csv` SHA-256: `D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B`
- `F_aia.csv` SHA-256: `6D4FF74F8FF2278E050A327F8982803F3CEBDFDEF5024F548FB1688FB9F761D6`

提出操作は行っていない。
