# 攻撃バージョン3: 分析・実験結果

実行日: 2026-09-03

## 結論

本命は次の組合せです。

- MIA: `v3_private_demotion_transport`
- AIA: `v3_retained_v2_adaptive`

MIAはv2順位から確定non-memberだけを降格するため、公式MIA ROCを標的ごとに非悪化にします。
AIAは追加案を採用せず、v2をそのまま維持します。

## 採点値の読み替え

| CodaBench表示 | 値 | 意味 |
|---|---:|---|
| 得点 | 0.28 | target別MIA最大値とAIA最大値の平均の和 |
| 内訳1 | 0.0867 | MIA全体 |
| 内訳2 | 0.3333 | MIA希少 |
| 内訳3 | 0.1171 | AIA全体 |
| 内訳4 | 0.1000 | AIA希少 |

内訳2は希少正例が存在するtargetだけのNaN除外平均です。このため、合計得点より大きくても
矛盾しません。また4内訳だけから、target別max後のMIA/AIA寄与を一意には復元できません。

## MIAの既知真値を使った改善

self challenge 326行のうち、公開QIで自分のBに完全一致した163行を一意に復元しました。
これはルール上、他targetと共有されない自チーム非共有ブロックです。

| 検査 | 結果 |
|---|---:|
| 候補行 | 1,049 |
| self challenge行 | 326 |
| 完全一致・確定non-member | 163 |
| うち希少行 | 13 |
| 残る攻撃可能候補 | 886 |

v2本命とv3本命の低FPR尾部を比較しました。

| 上位枠（20target合計） | v2の確定負例 | v3の確定負例 |
|---|---:|---:|
| top 10 | 14 | 0 |
| top 20 | 33 | 0 |
| top 50 | 134 | 0 |

残る886行のtarget内相対順位は全20列で完全一致します。希少top-1の確定負例は8/20targetから
0/20targetへ減りました。top-10の200枠はv3でも200候補すべて一意です。

## MIAの比較候補

- v1 transportは2seed shadowでraw`0.9938`、noise 3%`0.9925`、noise 10%`0.9722`、
  time shuffled`0.9747`、mixed fingerprint`0.9166`で安定していたため基礎順位に維持。
- DOMIAS追加は1seed比較の全条件で改善したため、本命生成でも維持。
- 希少専用並べ替えはraw rareが`1.0000 → 0.9848`、noise 10%でも`1.0000 → 0.9848`と
  下がったため不採用。
- defense fingerprintによるtransport隔離はmixedが`0.9166 → 0.9125`となり不採用。
- unconstrainedを10%混ぜる軽量blendはstress 3条件平均で約`+0.0027`でしたが、
  time shuffledで`-0.0014`かつ実戦真値なしのため、比較variantに限定。
- 886行だけで再transportする強い案も、残る正例順位を変えるため比較variantに限定。

## AIA比較

| 比較 | 結果 |
|---|---:|
| self既知member: v1 50%選別 | TP 84 / FP 79 |
| self既知member: 実験的密度選別 | TP 85 / FP 78 |
| self member時刻成功: v1 | 0.4356 |
| self member時刻成功: v2 | 0.4479 |
| 3 stress平均: 現行v2 | 0.8352 |
| 3 stress平均: fraction 0.40 | 0.7153 |
| 3 stress平均: rare patch最高 | 0.8308 |

実験的会員選別の改善は1行だけで、選択率変更と希少補正はshadowで低下しました。target 17は
event率100%という異常fingerprintを持つためhorizon-only案をvariantへ残しましたが、target別
公式真値がなく、v2も既に信頼距離を`0.08`へ抑制しているため本命は変更しません。

## 最終生成・検査

- 本命MIA: 1,049行 × 21列（record_id + 20target）
- 本命AIA: 6,520行（20target × 326 challenge）
- v1 15テスト、v2 13テスト、v3 9テスト: 全合格
- v3 strict preflight: 合格
- 公式MIA/AIA論理validator: 合格
- CSV再読込、主要範囲目視、エラー文字列scan: 合格

事務局限定のtarget別真値は手元にないため、v3の公式得点は提出前には算出できません。
