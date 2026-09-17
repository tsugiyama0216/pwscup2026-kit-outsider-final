# PWSCup2026 — 予備戦team24・本戦team10

予備戦の作業は **[`prelim/`](prelim/README.md)** に集約しました。本戦は [`main_round/`](main_round/README.md) で別管理します。

**本戦の現在の匿名化候補は[S2](main_round/anonymization/team10/S/S2/README.md)です。** S1の条件付きround処理をsampleへ修正し、ローカル有用性U=0.5484134662543412。学習済みDPモデル・検証記録・[提出用ZIP](main_round/anonymization/team10/S/S2/submission/team10_S2_defense.zip)をまとめています。S2のCodaBench提出は未実施です。

**9月9日の予備戦確定結果：team24の攻撃力は0.3406681446132233（4位）、総合得点は0.4223063973063974（17位）です。** [結果公開の整理・原本・訂正事項](prelim/results/20260909/README.md)を追加しました。本戦版ルールブック **main-process-20260912** の[原本と変更点](main_round/rulebook/README.md)も保存しています。

| 探したいもの | 場所 |
|---|---|
| 予備戦・匿名化フェーズの提出物 `C.csv` | [`prelim/anonymization/anonymization-submitted-v5/`](prelim/anonymization/anonymization-submitted-v5/) |
| 予備戦・攻撃フェーズの最終提出、候補、採点結果 | [`prelim/attack/`](prelim/attack/README.md) |
| 予備戦の実装、実験記録、当時のキット・ルール | [`prelim/workspace/`](prelim/workspace/README.md) |
| 初期の攻撃手法を説明した旧学習資料 | [`prelim/old_attack_study_materials/`](prelim/old_attack_study_materials/README.md) |
| 全体構造・旧パスからの移動先・実行方法 | [`REPOSITORY_LAYOUT.md`](REPOSITORY_LAYOUT.md) |

予備戦の最終攻撃は **v9 guarded、公式結果（ユーザー共有）0.3407** です。予備戦の提出枠は使い切っています。v7 は単独では未提出であり、本戦の性能も未評価です。

今回の整理は保存場所と案内の変更です。予備戦の作業環境は `prelim/workspace/` にまとめ、実装・データ・出力の相対配置を維持しています。過去の実行手順は、このディレクトリを作業場所として読んでください。攻撃アルゴリズムや提出 CSV の内容を更新する整理ではありません。

`prelim/workspace/participant_data/` は予備戦当時の**練習用データ**です。本戦データでも、実際に配布された team24 の B/B_self/A_bg でもありません。予備戦の実配布の生データは外部のローカル保存先に残し、新たなアップロードには含めません。実トークンも原則除外しますが、2026-09-17のユーザーの明示的指定により、本戦team10の `participant_data/token.txt` と `模擬提出データ/token.txt` は既存の非公開リポジトリで管理します。同日の追加許可により、[S1のトークン入り提出ZIP](main_round/anonymization/team10/S/S1/submission/team10_S1_defense.zip)も保存します。配置先は [`main_round/anonymization/team10/`](main_round/anonymization/team10/README.md) です。その他のトークン入りZIP、仮想環境・キャッシュは引き続き除外します（既存Git履歴は書き換えていません）。

本戦を始めるときは、新しいルール・配布物を確認してから `main_round/` に作業を追加します。予備戦の資料をそのまま本戦の仕様として扱わないでください。

2026-09-17のS2完了後の依頼により、S2と検証記録をmainへの反映対象としました。先に許可された秘密トークン入りZIPの取り扱いを引き継ぎ、S2の提出ZIPもこの既存の非公開リポジトリで管理します。
