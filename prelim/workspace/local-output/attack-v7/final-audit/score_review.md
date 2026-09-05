# PWSCUP2026 v7 最終 score_1 独立レビュー

## 結論

現行v7（共通0点7 targetの `complement-of-unanimous`、target 20の固定mask内7-cell
time decoder、MIA v4固定）は、既存の正常形式候補の中で最終採用してよい。

v6の方向的レンジ `0.335–0.352` に対する主推定の追加liftは `+0.020528`。
したがって、最終意思決定に使うv7の主レンジは **`0.356–0.373`** とする。
探索側からheld-outまでの研究間平均差をすべて含めた広い方向レンジは
**`0.351–0.374`**。いずれもprivate leaderboardの信頼区間や保証範囲ではない。

二重計上はない。7 targetのBoolean変更とtarget 20の時刻変更はtarget集合が排他的で、
両方ともv6 horizon-edgeを基準にした差分である。ただし、同じ変更を測った複数の
shadow結果を互いに足してはいけない。

## score_1 の再構成

採点式は次のとおり。

`score_1 = mean_target MIA_score_k + mean_target AIA_score_k`

AIAは20 targetの算術平均なので、7 targetの対象別平均差は `7/20`、target 20だけの
差は `1/20` を掛ける。

### 公式実測から固定できる基準点

| 成分 | 値 | 根拠 |
|---|---:|---|
| v4 MIA power | 0.157911487923 | `0.2771 - mean(independent aia_per_target)` |
| 公式target別winner AIA power | 0.150620412542 | v4 / independent / routed のtarget別最大値 |
| 固定基準 `S_known` | **0.308531900465** | 上記2成分の和 |

MIA値は表示されたscore_1からの逆算なので約`±0.00005`の丸め幅を持つ。v7の
`F_mia.csv`はv4とbyte-identicalであり、v7追加liftにMIA差はない。

既報のv6 `0.335–0.352` は、`S_known`にrouted完全補集合とhorizon edgeのローカル推定を
既に加えた方向レンジである。従ってv6で既に数えた推定増分は
`+0.026468–+0.043468`。v7ではこのレンジへ新しい**差分だけ**を加える。

### v7差分

| 変更 | 対象レベルの根拠値 | score_1換算 | 採用する値 |
|---|---:|---:|---|
| 7 zero targets: Boolean、locked fresh 30-seed | +0.044856945/target | `×7/20 = +0.015699931` | 主推定 |
| 7 zero targets: held-out exact-zero | +0.048263171/target | `×7/20 = +0.016892110` | 感度 |
| 7 zero targets: discovery exact-zero | +0.033725628/target | `×7/20 = +0.011803970` | 感度 |
| target 20: exact-production固定mask held-out | +0.096558421 | `/20 = +0.004827921` | 主推定 |
| target 20: discovery | +0.082223658 | `/20 = +0.004111183` | 感度 |

主計算は、strategy lock後のfresh Boolean確認とtarget 20のheld-out値を1回ずつ使う。

`0.015699931 + 0.004827921 = 0.020527852`

従って、

`[0.335, 0.352] + 0.020527852 = [0.355527852, 0.372527852]`

すなわち表示用には `0.356–0.373`。7 targetとtarget 20のdiscovery/held-out平均差の
端を組み合わせると `0.351–0.374` になる。

追加のall-rows-control fresh確認ではBoolean差が`+0.058722/target`、score_1換算
`+0.020553`だった。これは上側を支持する再現であり、主推定へさらに加算する別liftではない。

## 二重計上監査

1. **v6の完全補集合を再加算しない。** Boolean benchmarkのbaselineは
   `exact_complement_edge`。報告された`+0.044857/target`等は、v6が既に持つ163行の
   完全補集合に対し541行を追加した差だけである。
2. **horizon edgeを再加算しない。** Booleanの全active予測は9.5、target 20 benchmarkの
   baselineも9.5で、v6の`10.0 -> 9.5`利得は両benchmarkのbaseline側に入っている。
3. **target 20の既存scoreを再加算しない。** 公式winner合成にはtarget 20の
   `0.0520833`が既に含まれる。v7で加えるのは固定mask内7セルのtime差分だけである。
4. **反復推定値を合算しない。** Booleanのdiscovery、held-out、fresh 30-seed、
   all-rows-controlは同じ変更の推定。target 20の広域benchmarkと厳密7-cell再現も同じ変更の
   推定である。
5. **target間の算術加算は妥当。** score_1のAIA部分はtarget別scoreの平均で、
   Boolean対象 `{6,7,8,9,17,18,19}` とdecoder対象 `{20}` は排他的。各target内の
   `max(all, rare)`非線形性は既に対象別deltaの中で評価されている。

実ファイル比較でもv6→v7はAIA 548行だけが変わった。内訳は7 zero targetsの541行と
target 20の7行で、MIA差は0行だった。

## 下振れ感度

主レンジは期待方向の比較であり、下側保証ではない。

- Boolean fresh確認のp10は`-0.036399/target`、score_1換算で`-0.012739`。
- Booleanは公式集計だけでは符号を部分識別できず、MILPのraw advantageは
  `-418–+486`。採用根拠は条件付きshadowの再現である。
- target 20の機構保持held-outではp10がscore_1換算`+0.001675`、最小でも
  `+0.001316`だった。一方、event timeをQIから独立にshuffleするstressでは平均
  `-0.005872`相当となる。

これらは異なる標本・stress条件なのでquantile同士を足した正式な下限は作れない。
従って `0.356–0.373` を確率的信頼区間として説明してはならない。

## 未採用案の短い監査

「正常形式」「discoveryと未使用holdoutの分離」「公式結果との整合」「v7に対する
追加収益」の順で確認した。現行v7を変更するほど高収益でgateを通る案は残っていない。

| 案 | 既存証拠 | 監査判断 |
|---|---|---|
| target 20 trusted-link mask拡張（productionで4行追加） | 機構保持held-out 32/32非悪化。固定mask版への増分はscore_1約`+0.000351` | 正常形式・通常holdoutは通る唯一の未採用案だが、高収益ではなく新しいmembership仮説を増やすため据置 |
| target 19 q=0.5 event overlay | 機構保持validationはtarget差`+0.0985–+0.1110`（score_1約`+0.0049–+0.0056`） | time-shuffled validationで全件悪化、公式0点方向とも衝突し、現行Booleanのtarget 19と重なるため不採用 |
| positive targetsへのq=0.5 decoder追加 | target 20が平均を支配。残る候補5/12/26/29等の総追加は約`+0.0006–+0.0011`、一部負例あり | 高収益・下側gateを満たさない |
| all-rows edge | fresh exact-zeroで完全補集合比score_1約`+0.00949` | 同じ標本でBooleanは約`+0.02055`。代替案であって追加案ではなく、Booleanより弱い |
| partial active-count / ranking tail | 最良held-outは`+0.000415/target`、勝率46.2% | score_1換算約`+0.000145`でgate失敗 |
| pair complement / majority / `000`限定 | Booleanより一貫して弱い | 代替であり非加算、不採用 |
| MIA Daybreak-v4 rare lock等 | 正常形式だがheld-out power差`-0.052849`、0/12勝 | MIA v4固定を維持 |
| AIA rare proxy | 公開背景でrare分類性能は高いが、提出攻撃としての独立holdout liftがない | production変更のgate未到達 |
| cross-target exact reuse | challenge間・releaseとのexact一致0 | 利用可能な正常形式案なし |

## QAと判定

- v7 AIA: 6,520行、6,520 unique `(target_k, challenge_row_id)`
- v6→v7 changed rows: 548、変更targetは6/7/8/9/17/18/19/20のみ
- v7 MIA: v6/v4と同一
- v7 `F_aia.csv` SHA-256: `65AD898E8E24497AA1C415BBCBE14D8360BC35676903A069162F4F20F115EAA0`
- focused tests: 5/5 PASS
- official logical validator: PASS（既存diagnosticsおよびfocused testで再確認）
- production candidate、ZIP、token、提出操作: 本監査では変更なし

総合評価は **Share with caveats / 最終採用可**。最終説明では主レンジ
`0.356–0.373`、広い研究間レンジ`0.351–0.374`、および「private保証ではない」を
セットで示すのが妥当である。

## 参照した既存成果物

- `local-output/attack-v6/FINAL_LOCAL_REEVALUATION.md`
- `local-output/attack-v6/final_feedback_winners.json`
- `local-output/attack-v7/aia-official-bayes/final_summary.json`
- `local-output/attack-v7/active-fraction/confirmation_summary.json`
- `local-output/attack-v7/active-fraction/all_rows_control_summary.json`
- `local-output/attack-v7/aia-time-decoder/REPORT.md`
- `local-output/attack-v7/aia-time-decoder/target20_membership_expansion_report.json`
- `local-output/attack-v7/mia-lowfpr/final_summary.json`
- `local-output/attack-v7/aia-rare/rare_proxy_validation.json`
- `local-output/attack-v7/cross-target-reuse/summary.json`

As of: 2026-09-05 Asia/Tokyo
