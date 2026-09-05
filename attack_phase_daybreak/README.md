# PWS Cup 2026 Daybreak Blue 攻撃フロー

このディレクトリは、PWSCup2026予備戦team24の配布済みローカルCSVだけを対象にした、
Daybreak Blue監査版のMIA/AIA生成フローです。ネットワーク接続、token読取、ZIP作成、提出操作は
実装していません。

## 本命選択

Daybreak Blueでv1〜v3の実装、shadow結果、自己チャレンジ既知ラベル、公開Cのfingerprintを
独立に監査しました。本命は次を維持します。

- MIA: `v3_private_demotion_transport`
- AIA: `v3_retained_v2_adaptive`

MIAはルールブック§3.1/§4.2に基づき、自己AIA challengeから一意に照合できる非共有ブロック
163行を、他20targetの全列で最下位へ移します。残る886行のtarget内順位は維持するため、
ルール前提の下でROCを非悪化にできます。

AIAのtarget17 horizon-only、C18/C29の背景donor除外、C19/C29の重複除外などは比較案としては
合理的ですが、target別公式真値がなく非悪化保証もないため本命へ入れていません。

## v3から追加した安全策

- `SHA256SUMS.txt`記載ファイルを全件検証してから入力を読む
- team24の予備戦20targetを固定し、練習用29targetとの取り違えを停止
- MIA共有総数を正しい`508`に固定（v3 runner既定`736`の指定漏れを防止）
- 非共有ブロック規則の確認フラグを必須化
- 公式validatorに加え、`inf`、MIA tie、AIA複合キー重複を独自検査
- 書出し後のCSVを再読込して同じ検査を再実行
- candidateディレクトリには`F_mia.csv`と`F_aia.csv`以外を許可しない

## 実行

リポジトリルートで次を実行します。引数なしで、このワークスペースのteam24予備戦パスを使います。

```powershell
powershell -ExecutionPolicy Bypass -File .\attack_phase_daybreak\run_daybreak.ps1
```

通常はDOMIAS込みで実行してください。`-SkipDomias`は高速な動作確認専用です。

出力先:

- 検証・比較: `local-output/prelim/team24/attack-daybreak-blue/`
- 提出候補CSV 2枚のみ: `submissions/prelim/team24/attack-candidate-daybreak-blue/`

公式のtarget別membership/AIA真値はローカルにないため、公式攻撃得点は提出前には算出できません。
このフロー自体は提出を行いません。

## 生成結果（2026-09-03）

- `F_mia.csv`: 1,049行 × 21列、SHA-256
  `8236A9A572306C79BE2D417FCA8F6BA395BCA1F93BA3E90DC99642A2010A4BB2`
- `F_aia.csv`: 6,520行 × 3列、SHA-256
  `340BD8DD0571A15C52FA30B6328265450DBDC2E5689DD51934C28EC3F833FE20`
- 配布manifest 44ファイル: 全件一致
- v1 15件、v2 13件、v3 9件、Daybreak 6件（計43テスト）: 全合格
- 独自strict preflight、公式MIA/AIA logical validator、書出し後再読込: 全合格

Daybreak Blueの本命判断がv3維持だったため、2枚のCSVは検証済みv3本命とbyte-for-byteで同一です。
これは単なるコピーではなく、配布入力からDOMIAS込みで再生成した結果です。
