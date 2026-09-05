# 予備戦の攻撃ファイル

最終提出は **v9 guarded / 公式得点 0.3407** です。予備戦は終了しています。

```text
prelim/attack/
├─ final/        最終提出版 v9 guarded のCSV 2本
├─ candidates/   それ以前の17候補のCSVペア
├─ results/      公式結果JSON 6本と最終結果レポート
├─ manifest.json コピー元・サイズ・SHA-256
└─ README.md     この案内とコード・研究記録へのリンク
```

## 最終提出ファイル

- [F_mia.csv](final/F_mia.csv)
- [F_aia.csv](final/F_aia.csv)
- [最終公式結果JSON](results/guarded_final.json)
- [最終結果レポート](results/FINAL_RESULT.md)

この2つのCSVは提出済みv9 guardedの原本と**バイト単位で同一**です。
token.txtやトークン入り提出ZIPは、この整理フォルダには含めていません。

## 過去の候補と結果

[candidates/](candidates/) は17候補を元の候補名ごとにまとめています。
各フォルダにはF_mia.csvとF_aia.csvだけを置きました。提出済みだけでなく、
未提出の実験候補も含まれます。候補名だけで採点済みとは判断しないでください。

[results/](results/) にはv3、v4、Daybreak、independent probe、routed-guided probe、
最終v9のユーザー共有結果を保存しています。
レポート中の実装参照は[元のレポート](../../attack_phase_v9/FINAL_RESULT.md)の位置を基準とします。

## 攻撃コードと研究記録

既存のimport・データ参照を保つため、実行コードと研究ログは移動せず、
この場所から辿れるようにしています。

- 初期実装：[v1](../../attack_phase/)・[v2](../../attack_phase_v2/)・[v3](../../attack_phase_v3/)・[v4](../../attack_phase_v4/)
- 後期実装：[v5](../../attack_phase_v5/)・[v6](../../attack_phase_v6/)・[v7](../../attack_phase_v7/)・[v8](../../attack_phase_v8/)・[v9](../../attack_phase_v9/)
- 別系統：[Daybreak](../../attack_phase_daybreak/)・[Daybreak independent](../../attack_phase_daybreak_independent/)
- [全研究ログ・検証出力](../../local-output/)（不採用実験を含む）
- [全作業アーカイブの説明](../../research_archive/README.md)
- [本戦へ再開する際の引き継ぎ](../../CODEX_HANDOFF.md)

元の提出CSV・結果ファイルも残しています。このフォルダは予備戦の固定コピーで、
本戦用のファイルや加工・防御用C.csvとは混ぜません。
