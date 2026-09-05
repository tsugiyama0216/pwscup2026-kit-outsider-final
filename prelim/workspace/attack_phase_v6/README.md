# 攻撃バージョン6

v6は、チーム自身に返された公式feedback、主催者配布データ、ローカルshadowだけを
使う残り2回向けの攻撃です。次回は情報価値を持つ強化候補を提出し、そのtarget別結果を
使って最終回のAIA block winnerとMIA方式を選びます。CodaBenchへの提出は自動化しません。

## 公式結果の読み取り

| MIA | all | rare | AUC |
|---|---:|---:|---:|
| v3 | 0.0926638 | 0.3333333 | 0.6434884 |
| v4 / 直近probe | 0.1023675 | 0.3333333 | 0.6493386 |
| Daybreak independent | **0.1062194** | 0.1666667 | 0.6356009 |

Daybreak MIAはallだけがv4より高く、rareとAUCは低下しました。したがって全面置換は
行わず、v4の上位tailで同じrecordが複数targetに現れる場合の所有先判断だけに使います。

公式`aia_per_target`から厳密に計算できるAIA powerは次の通りです。

| AIA | power |
|---|---:|
| v4 | 0.1340658382149063 |
| independent | 0.1191885120769747 |
| target別winner合成 | **0.1472268438242094** |

winner合成は各targetの326行を丸ごとコピーするため、v4比
`+0.0131610056093031`は公式結果から確定します。これは次回routeの得点予測ではなく、
最終回で必ず利用できる既知の下地です。

## 次回提出の本命

本命は`attack-candidate-v6-routed-guided-probe`です。

### AIA route

| target | variant | 根拠 |
|---|---|---|
| 1, 2 | exact v4 | 公式target scoreが既に1.0 |
| 3 | equal consensus + copula mix + independent/event | shadowで安定した局所改良 |
| 4, 5, 12 | equal copula + v4/event | v4の正の時刻情報を優先 |
| 10 | v4-lean copula + v4/event | independentが公式で弱いtargetを保護 |
| 29 | exact v4 | shadow評価版と生成版の差を検知したため、公式実測winnerを保護 |
| 13, 22, 23 | equal copula + independent/event | independentの公式優位を利用 |
| 26 | soft copula + independent/event | independent優位とcopula近接を強める |
| 6, 8, 17 | existing copula selection + horizon | 密度方式のtarget固有gateを通らない |
| 7, 20 | global density all5 + horizon | 20表差替え負対照でtarget固有、6/6 split勝利 |
| 9 | global density stable3 + horizon | age/SBP/TGが最も安定、6/6 split勝利 |
| 18 | sex consensus noHDL + horizon | prefecture破壊に対しsex関係だけが強い |
| 19 | **inverted global density stable3 + horizon** | 公開表がほぼA_bgなので通常の向きが逆 |

positive targetのevent-time置換は、sex/prefecture内の相互最近傍かつ近傍margin上位半分の
公開eventだけに限定します。既存攻撃が0点のtargetでは個別時刻を移植せず、selectedを
`10.0`、unselectedを`11.0`とします。

target 29は当初のshadow選定版とfilter適用後の生成版で9行が一致しなかったため、未検証の
生成blockを採用せず、公式で`0.0762712`を得たv4 blockへ戻しました。残るpositive 9 targetは
shadowで評価したproduction blockと一致しています。

target 19では、C19の`1028/1049`行がA_bgと`age/SBP/TG`で完全一致します。
標準filter後も`979/999`行が一致します。通常密度が選ぶ既知control率はproduction互換
6 splitすべてで0.5超、平均`0.580860`、20個のC差替えでもC19が1位でした。そのため
target 19だけdensity順位を反転します。generatorは一致率90%以上を実行時gateにします。

### MIA guided equal-rank tail15

v4 top-15の300枠には16個の重複枠があります。まず保持数を理論最大の284/300に保った
まま重複をゼロにし、重複recordの候補owner間だけで
`v4 rank + Daybreak rank`を最小化します。v3は最終tie-breakと補充にだけ使います。

この規則でownerが変わるのは5件です。

| record | owner変更 |
|---:|---|
| 410 | 29 → 8 |
| 647 | 10 → 29 |
| 807 | 9 → 6 |
| 998 | 5 → 22 |
| 1001 | 3 → 20 |

次の不変条件をテストで固定しています。

- target 1/2の全順位はv4と同一
- 既知private bottom 163の集合と順位は全targetで同一
- eligible rare 11行とprivate bottomの順位をv4どおり保つため、公式rareは厳密に`1/3`のまま
- top-15は300 recordすべて一意
- v4からのKendall inversionは全20列合計98だけ

Daybreakのall改善はtarget 1/2の各1 TPだけでも説明できるため、この5-switchの改善は
保証されません。直接ownerを5件とも誤る最悪方向はMIA-all平均で約`-0.0093〜-0.0100`
です。だからこれは残り2回のうち次回だけで測る仮説であり、未測定の最終一発には使いません。

## 次回feedbackの判定gate

次回は必ず総合score、MIA all/rare/AUC、20個の`aia_per_target`を保存します。

1. `new MIA power = score_1 - mean(aia_per_target)`を計算する。
2. 比較baselineは直近probeから得た`0.1579114879230253`（`0.2771 - 0.1191885120769747`）。
   `score_1`表示丸めによりbaseline自体に約`±0.00005`の幅がある。
3. 差が丸め誤差を十分超えて高ければguidedを最終回でも維持する。
4. 低ければv4 MIAへ戻す。同値域でも、公式実測とAUCで優るv4を選ぶ。
5. AIAはv4、independent、次回routeのtarget別実測winnerを326行単位で合成する。

guided MIAのsanity checkは、rare=`0.3333333`、AUCが
`[0.6491472063, 0.6495300188]`内です。外れた場合は提出ファイルまたは結果の紐付けを
疑います。

## 生成物

| 用途 | ZIP | SHA-256 |
|---|---|---|
| **次回本命** | `local-output/attack-v6/attack-v6-routed-guided-probe.zip` | `64B845B5EB17954B4DD52A264B28733E190CC998CBDF5FB1880558464990657A` |
| MIA即時切戻し版 | `local-output/attack-v6/attack-v6-routed-v4-probe.zip` | `3D2D5967055D3134E67A30A32C93EEC51427B70C30746821DECFEFD8D97A9FDB` |

本命CSV hashは以下です。

- `F_mia.csv`: `70E8E689E476E9EEF54F5821846EC1B93DE04FD6A63B680849D34F720DB78AFA`
- `F_aia.csv`: `CCAA0A4EE58797F263E1281D5BC65E763B129A2F77DE00A1A4579DD60566129F`

両ZIPとも主催者の公式package validatorを通過済みです。ZIPにはtokenが入るため
`local-output`から外へ移さず、Gitへ追加しません。

## 最終回: 公式0点blockの完全反転

直前提出の公式結果は総合`0.3004`、MIA all/rare=`0.1024/0.3333`、
AIA all/rare=`0.1256/0.1000`でした。guided MIAの推定寄与はv4との差が
`+0.00000736`だけで表示丸め幅内なので、最終版はMIAをbyte-identicalなv4へ戻します。

AIAはまず3回の公式実測からtarget別winnerを丸ごと合成します。この既知部分のpowerは
`0.1506204125415067`で、MIAとの再構成総合値は約`0.3085`です。その上で、3候補すべてが
公式0点だった`6, 7, 8, 9, 17, 18, 19`だけ、直前routed blockの二値予測を全行で
`10.0 <-> 11.0`反転します。13個の正点blockは実測winnerと完全同一です。

各target scoreは0以上へclipされるため、この探索で既知の正点寄与を失いません。さらに
直前routedの7 targetは公式集計から、10年予測の成功数が会員側より対照側で合計
`59〜68`件多かったと制約できます。対照は会員と公開stratumおよびonsetでmatchingされるため、
補集合は未実測のtree/KNNより公式の負方向を直接利用する最終仮説です。ただし0.5到達には
7 targetから合計約`3.83`（平均約`0.547`）の追加powerが必要であり、到達保証ではありません。

最終提出用ZIP:

- `local-output/attack-v6/attack-v6-final-aggressive-complements.zip`
- ZIP SHA-256: `11027B5838D92E2696DE11185BB2BC841CA701A710DD1E77F9C2D5957BE5357D`
- `F_mia.csv`: `D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B`
- `F_aia.csv`: `2CFDC03D9EB2A2F096BE3A872BC836CA0AAE16FF170F20F0124F85F14E7A3D82`

公式package validator、全110テスト、`compileall`を通過済みです。提出操作は行いません。

## 最終ローカル再評価: horizon edge

上記のmember選別、event時刻、`11.0` abstainを固定したまま、target 1/2を除く
exact `time_hat=10.0`の2,685行だけを`9.5`へ移した候補を追加しました。真値の許容域
`0 < time <= 10`と`tau_time=0.5`の下で、行単位の成功区間は`[9.5, 10]`から
`[9.0, 10]`へ広がり、従来の成功行を失いません。

24 paired shadowでは、公式正点11 targetの総合換算平均が`+0.0024505`でした。
公式0点blockを再現できたtarget 19も24/24で非悪化し、公開時刻分布による7 blockの
追加見込みを合わせると、完全補集合版から約`+0.0051〜+0.0068`がローカル期待値です。
条件付きの総合レンジは概ね`0.335〜0.352`であり、private真値に対する保証値ではありません。
rare-top26、age近傍pair、event-q75移植、inverse consensusはいずれもholdout、下側感度、
または公式制約で棄却しました。

最終推奨ZIP:

- `local-output/attack-v6/attack-v6-final-horizon-edge.zip`
- ZIP SHA-256: `D0E93343805D2A303EA752FA4CF1E176DCFE80CD56F54488B3DD36C38ACBD1E6`
- `F_mia.csv`: `D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B`
- `F_aia.csv`: `6D4FF74F8FF2278E050A327F8982803F3CEBDFDEF5024F548FB1688FB9F761D6`

公式package validatorと全113テストを通過済みです。提出操作は行いません。

## 再現

```powershell
$python = "C:\Users\takut\.codex\.chatgpt-projects\g-p-6a853f607498819195a7ef70756f71c9\pwscup2026-kit-private\starter\.venv\Scripts\python.exe"
& $python -m attack_phase_v6.generate_submission_v6
& $python -m unittest discover -v
& $python -m attack_phase_v5.package_submission_v5 `
  --candidate-dir .\submissions\prelim\team24\attack-candidate-v6-routed-guided-probe `
  --out .\local-output\attack-v6\attack-v6-routed-guided-probe.zip

# 最終版
& $python -m attack_phase_v6.generate_final_submission_v6
& $python -m attack_phase_v6.generate_aggressive_final_submission_v6
& $python -m attack_phase_v5.package_submission_v5 `
  --candidate-dir .\submissions\prelim\team24\attack-candidate-v6-final-aggressive-complements `
  --out .\local-output\attack-v6\attack-v6-final-aggressive-complements.zip

# 最終ローカル再評価版
& $python -m attack_phase_v6.generate_horizon_edge_final_submission_v6
& $python -m attack_phase_v5.package_submission_v5 `
  --candidate-dir .\submissions\prelim\team24\attack-candidate-v6-final-horizon-edge `
  --out .\local-output\attack-v6\attack-v6-final-horizon-edge.zip
```

generatorはtokenを読みません。packagerだけが既存tokenを読み、表示せずにZIP内CSV hashと
候補CSV hashの一致、flatな3ファイル構成、公式validator通過を確認します。0.4到達は
ローカルから保証できず、shadow値は方式選択の方向的証拠としてのみ扱います。
