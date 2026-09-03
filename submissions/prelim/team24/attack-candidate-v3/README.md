# 予備戦・攻撃フェーズ提出候補 v3（team24）

2026-09-03時点のローカル生成・検証済み候補です。`attack_phase_v3/`から生成し、従来の
`attack-candidate-v1/`、`attack-candidate-v2/`とは別に保存しています。提出操作は行っていません。

## v3の本命変更

- MIA: self challengeから特定した、他targetへの確定non-member 163行を全20列の最下位へ移動
- MIA: 残る886行のv2内相対順位と、低FPR域の標的横断排他性は完全維持
- MIA希少: 24希少行中13行が確定non-member。希少top-1の確定負例を8/20targetから0へ削減
- AIA: 追加案がself既知ラベルとshadow stressでv2を上回らなかったため、v2本命を意図的に維持

v2では確定負例が全target合計でtop-10に14件、top-20に33件、top-50に134件ありました。
v3ではすべて0件です。このMIA変換は正例を降格せず、確定負例だけを下げます。

## 収録ファイル

- `F_mia.csv`: 1,049行、`record_id`と攻撃相手20targetの計21列
- `F_aia.csv`: 6,520行、20target × 326 challenge行

自チーム`24`はMIA/AIAとも除外しています。`token.txt`は読み取らず、このフォルダにも複製して
いません。提出ZIPにはこのREADMEを含めず、CSV 2枚とユーザー管理の`token.txt`だけを直下へ
入れてください。

## 検証結果

- v3 strict preflight: MIA ID/対象列/有限範囲/tie-free、AIA複合キー/全ID/有限非負を確認
- キット公式MIA/AIA論理validator: 合格
- v1 15テスト、v2 13テスト、v3 9テスト: 全合格
- CSV再読込と先頭範囲の目視確認: 合格

SHA-256:

- `F_mia.csv`: `8236A9A572306C79BE2D417FCA8F6BA395BCA1F93BA3E90DC99642A2010A4BB2`
- `F_aia.csv`: `340BD8DD0571A15C52FA30B6328265450DBDC2E5689DD51934C28EC3F833FE20`

`F_aia.csv`がv2と同じhashなのは意図した結果です。事務局限定真値がないため、v3の公式得点は
提出前には算出できません。
