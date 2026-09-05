# PWSCUP2026 MIA low-FPR v7 research

## 結論

**最終提出のMIAはv4を維持する。** 研究候補
`daybreak_v4_rarelock` は公式allチャネルだけを見ると有望だが、採点対象の
`mean_target max(TPR_all, TPR_rare)` で再現優位を確認できなかった。従って最終一回で
差し替えるためのgateを通していない。

## 研究候補

- path: `C:\Users\takut\Documents\ChatGPT\PWSCUP2026\pwscup2026-kit-private\local-output\attack-v7\mia-lowfpr\production_variants\F_mia_daybreak_v4_rarelock.csv`
- SHA-256: `D5285EF7F57AB743665284CEE666179185DE84D52B5CEC353DFAEFD5E1C6758F`
- v4からの変更: 17,631 cells / 886 distinct rows / 20 targets
- Daybreakからの変更: 179 cells（eligible rare 11行の列内順位修復のみ）
- 正常形式: 1,049行、20 target、finite [0,1]、全列tieなし

方式はDaybreak-independentの各target列を土台にし、既にv3/v4から回収済みの
eligible rare 11行だけを、Daybreakの同じ11 score slot内でv4順に並べ替える。
private 163行のbottom集合は維持する。LBから個別会員/非会員を断定したり、個別行を
正解扱いしてspliceしたりしていない。

## 公式結果からの強い事実

- Daybreak all=`0.1062193732` はv4 all=`0.1023675214`より `+0.0038518519`。
- rare順序はv4と完全同一になるため、candidate rareはv4と同じ `1/3`。
- rare-lockがDaybreakの低FPR prefixを変え得るのはtarget 19のrank 12だけ。
  target 1のrank 7/25 swapはband 4の最小prefix内に両方入る。
- よってcandidate allは保守的に `[0.1042193732, 0.1082193732]`、
  v4比下限 `+0.0018518519`。

ただしこの下限はallのtarget平均であり、採点値のtargetごとの
`max(all, rare)`には直結しない。v4 rareが勝つtarget上のall改善は得点に寄与せず、
別targetのall悪化だけが残る場合がある。

## discovery / held-out

固定recipeをdiscovery 4 seed × 2 scenario、held-out 6 seed × 2 scenarioで評価した。
validationはrecipe選択に使用していない。

- discovery delta power mean: `-0.045577` (8 observations)
- held-out delta power mean: `-0.052849`
- held-out median: `-0.047088`
- held-out p10: `-0.089380`
- held-out minimum: `-0.096632`
- held-out win rate: `0.0%` (0/12)
- held-out rare delta minimum: `+0.000000`（rare不変は確認）

toy shadowはproduction Bと同一ではないためLB予測ではないが、下振れ方向が12/12で
一貫しており、差替えを支持する再現証拠にはならない。

## feedback-constrained bound

v3/v4/Daybreak/guidedのall集約、全target band、target size 13×25+6×26+1×27、
排他性、rare hit比、v4/guidedの総合score表示幅を同時制約した7,010変数MILPを実行した。
両側120秒でfeasible incumbentなしのtime limitとなり、厳密power保証は得られなかった。
LP relaxationは `[-0.157960, +0.842137]` と緩すぎ、候補の非劣性を証明しない。
これは個別行真値の復元・断定には使っていない。

## validation

- official `validate_mia_submission` layer-2（distなし）: PASS
- production-reference strict validator: PASS
- focused unit tests: 5/5 PASS
- repository `unittest discover`: 118/118 PASS
- full ZIP/package validator: 未実行（production/ZIP/token非変更という本タスク範囲を遵守）

## 採否

公式allだけなら改善する巧い合成だが、最終scoreのworst-case保証がなく、held-outは
明確に悪化した。最後の一回ではv4から変更しない。
