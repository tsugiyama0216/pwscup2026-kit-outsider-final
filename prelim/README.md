# 予備戦アーカイブ

**[2026-09-09 結果公開の整理](results/20260909/README.md)**：team24は攻撃力4位・総合17位。全チームの確定値、攻撃力の訂正、公開された原本一式と検証記録を保存しています。過去の0.3407は攻撃力の丸め値です。

このディレクトリ以下は、予備戦で使用・検討した提出物、実装、実験、配布キット、資料です。本戦はリポジトリ直下の [`main_round/`](../main_round/README.md) に分けます。

| ディレクトリ | 役割 |
|---|---|
| [`anonymization/`](anonymization/README.md) | 匿名化フェーズの提出物。`anonymization-submitted-v5/C.csv` |
| [`attack/`](attack/README.md) | 攻撃フェーズの引き継ぎ用ファイル集。最終 v9、過去候補、公式結果 |
| [`old_attack_study_materials/`](old_attack_study_materials/README.md) | 初期の攻撃手法を扱う旧学習資料。最終 v9 の仕様書ではありません |
| [`workspace/`](workspace/README.md) | 予備戦の再現用作業環境。コード、元の候補、実験記録、当時のキット・ルール |

最終攻撃の公式結果（ユーザー共有）は **v9 guarded：0.3407**。予備戦の提出枠は残っていません。詳細は [`attack/results/FINAL_RESULT.md`](attack/results/FINAL_RESULT.md) を参照してください。

`attack/` は確認しやすい提出ファイル集で、`workspace/submissions/prelim/team24/` に再現用の元候補を残しています。別々の提出結果ではありません。対応とハッシュは [`attack/manifest.json`](attack/manifest.json) で確認できます。

実装を実行するときは `prelim/workspace/` を作業ディレクトリにします。予備戦当時の相対配置を保つため、攻撃コードと匿名化コード（`opendp_aim/` など）は、この作業環境内にまとめて保存しています。`participant_data/` は練習用であり、実配布の B/B_self/A_bg はリポジトリ外です。

全体のツリーと旧パスの移動先は [`REPOSITORY_LAYOUT.md`](../REPOSITORY_LAYOUT.md) にあります。
