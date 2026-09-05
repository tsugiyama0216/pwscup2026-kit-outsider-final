# v7 最終独立監査

監査日: 2026-09-05 (JST)

## 結論

**CSV候補としては Ready to share。阻害事項なし。** 現在の
`attack-candidate-v7-feedback-stack` は、公式logical validatorを通過し、全20 targetの
キー集合、行数、複合キー一意性、入力・出力SHA256が一致した。production、ZIP、token、
提出処理には触れていない。

ただし、これは形式・集計・不変条件の監査であり、未知真値に対する攻撃力の優位性を保証する
ものではない。性能面の採否判断を上書きしない。

## 4項目の回答

### 1. AIAで163件を超えるactive予測は許容されるか

**許容される。** `active` は公式CSVのフィールドでも公式採点器の状態でもなく、本プロジェクトが
`time_hat != 11` を数えるために用いた戦略上の呼称である。公式validatorは固定3列、
`time_hat` の数値性・非負性、target集合、challenge ID集合を検査するが、163件制約を持たない
（`starter/src/pwscup2026/kit/validate.py:217-259`）。

実候補は公式validatorでPASS。さらに、メモリ上だけで全20 targetを326 activeに変えた
negative-controlもPASSした。v7で163件を超えるtargetは次の7つ。

| target | active |
|---:|---:|
| 6 | 217 |
| 7 | 253 |
| 8 | 211 |
| 9 | 274 |
| 17 | 223 |
| 18 | 236 |
| 19 | 268 |

### 2. `time_hat=11` はどう扱われるか

**正式な棄権分岐ではなく、必ず失敗する通常の数値予測である。** 公式validatorは11を有効な
非負値として受理する。採点は各行を除外せず、
`abs(time_hat - truth_time) <= 0.5` の真偽をmember/control各群の分母に含める
（`starter/src/pwscup2026/scoring/attack_score.py:155-156`）。真値の正式範囲は
`0 < time <= 10`（`starter/src/pwscup2026/common/submission_schema.py:194-200`）なので、
11の最小誤差は1.0年となり、許容誤差0.5年を必ず超える。

したがって「棄権」は正確には **validatorに許容された guaranteed-miss sentinel** である。
行は捨てられず、その実際のmember/control群で失敗1件として効く。

### 3. `score_1` の20-target平均換算

20 targetが全て存在する現候補では、正確な式は次のとおり。

```text
m_k = max(TPR_all_k, TPR_rare_k)                     # MIA target k
a_k = max(R_time_all_k, R_time_rare_k_after_deadzone) # AIA target k

score_1 = mean_k(m_k) + mean_k(a_k)
        = (sum_k m_k + sum_k a_k) / 20
```

根拠はMIAのtarget内max（`attack_score.py:66`）、MIAのtarget平均
（`scoring_io.py:280`）、AIAのtarget内max（`attack_score.py:177`）、AIAのtarget平均
（`scoring_io.py:325`）、両powerの加算（`scoring_io.py:349`）。

よって、他target不変なら1 targetのAIA score変化 `Delta` は `score_1` に
`Delta / 20 = 0.05 * Delta` 寄与する。v7はMIAがv4とbyte-identicalなので、

```text
score_1(v7) - score_1(base)
  = sum_k [a_k(v7) - a_k(base)] / 20
```

である。なおLBの `score_2`～`score_5` は各チャネルを別々にtarget平均した表示値であり、
それらだけからtarget内maxを含む `score_1` を厳密復元できない。

### 4. candidate key/hash不変条件

全てPASSした。

- 候補直下は `F_mia.csv` と `F_aia.csv` の2ファイルだけ。
- `F_mia.csv`: 1,049行、v4およびv6 baseとbyte-identical。
- `F_aia.csv`: 6,520行 = 20 target x 326行、固定3列、複合キー重複0。
- 各targetのchallenge ID集合は配布物と完全一致し、AIAキー列と行順もv6 baseと完全一致。
- 値が変わったtargetは `6,7,8,9,17,18,19,20` のみ。base比の変更行数は順に
  `54,90,48,111,60,73,105,7`。
- generatorの9入力SHAロックが全一致。配布manifest 44項目も全一致。
- 公式MIA/AIA logical validator、v7 strict variable-cardinality検査ともPASS。

固定出力SHA256:

```text
F_mia.csv D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B
F_aia.csv 65AD898E8E24497AA1C415BBCBE14D8360BC35676903A069162F4F20F115EAA0
```

generatorは入力SHAを読込前に検査し（`generate_submission_v7.py:377`）、キー不変を検査
（同`:331`）、書込前・書込後の出力SHAを検査する（同`:430-442`）。

## 注意点（現候補への影響なし）

公式AIA logical validator単体には次の穴があり、実測negative-controlでも受理された。

- 同じ複合キーの余分な1行を追加しても、IDをset比較するためPASSする。
- `time_hat=+inf` を有限性違反としてrejectしない。
- 整数と同じ値へ切り捨てられる小数 `challenge_row_id` をPASSさせる。

現候補では `_normalized_aia`、`validate_variable_aia`、配布キーとの326行完全一致、および
出力ファイル全体のSHAロックがこれらを全て遮断しているため、**現在の固定byte列には影響しない**。
運用上は `--skip-output-hash-gate` を使わず、package化後にも上記2 SHAを再確認すること。

候補directory自体には意図どおり `token.txt` がなく、提出packageではない。この監査では
ZIP/token/productionを対象外としたため、最終packageについては別途、CSV SHA再照合後に
`validate_submission_zip` を通す必要がある。

## 再現物

- `local-output/attack-v7/final-audit/audit_v7.py`: 読み取り専用監査コード
- `local-output/attack-v7/final-audit/audit.json`: 全チェック、target別行数・active数、
  negative-control結果

