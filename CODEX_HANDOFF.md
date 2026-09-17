# PWSCUP2026 引き継ぎ

予備戦は **v9 guarded、確定攻撃力0.3406681446132233（4位）** で終了しました。過去の0.3407は攻撃力の表示丸めです。**総合得点は0.4223063973063974（17位）**。提出回数は残っていません。
9月9日の[公開結果・原本・照合記録](prelim/results/20260909/README.md)を追加しました。公開AIAの攻撃者4→標的23の欠落と平均分母の要確認点も記録しています。team24の公開攻撃CSVは保存済み最終v9と完全一致しています。

本戦版[main-process-20260912のPDFと変更点](main_round/rulebook/README.md)は保存済みです。**2026-09-16に、本戦・加工フェーズのteam10配布データを[main_round/anonymization/team10/](main_round/anonymization/team10/README.md)へ配置しました。** 予備戦のteam24とはコホート番号が異なります。本戦の採点キットの導入・検証、加工実装、提出はまだ行っていません。

- [リポジトリ構造と旧パス対応](REPOSITORY_LAYOUT.md)
- [予備戦の全体案内](prelim/README.md)
- [最終攻撃CSVと公式結果](prelim/attack/README.md)
- [匿名化フェーズの提出物](prelim/anonymization/README.md)
- [詳細な研究引き継ぎ](prelim/workspace/CODEX_HANDOFF.md)
- [最終実測結果の説明](prelim/workspace/attack_phase_v9/FINAL_RESULT.md)
- [本戦の格納先](main_round/README.md)
- [本戦・加工フェーズのteam10データ](main_round/anonymization/team10/README.md)

旧ルートの実行コード・データ・研究記録は、相対配置を保って `prelim/workspace/` に移動しました。
過去の `python -m attack_phase_...` コマンドは、先に `cd prelim/workspace` して実行します。
詳細引き継ぎに残る過去の提出推奨・残回数は履歴であり、現在の指示ではありません。
旧学習資料は初期v1の説明で、最終v9の仕様書ではありません。

本戦・加工フェーズの配布原本6ファイルは、2026-09-16のユーザーの明示的依頼に基づき、既存の非公開GitHubリポジトリへの保存対象としました。2026-09-17には、team10の `participant_data/token.txt` と `模擬提出データ/token.txt` も同じ非公開リポジトリへ保存するよう明示的に指定されました（いずれも `main_round/anonymization/team10/` 配下）。同日の追加許可により、`S/S1/submission/team10_S1_defense.zip` も同じ非公開リポジトリへの保存対象です。その他の実トークン・トークン入りZIP・予備戦の実配布B/B_self/A_bgは新たにアップロードしません。トークンの内容は文書やログに記載しません。
既存の非公開Git履歴を残すという所有者の判断は維持し、履歴書き換えやforce pushはしません。
