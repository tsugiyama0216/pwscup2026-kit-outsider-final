# 本戦

**模擬問題の生成と攻撃手法の比較は [mock_test/](mock_test/README.md) から始めます。** 複数抽出を保存してM・TSの手法を同じ問題へ適用し、平均に加えて問題別・加工方式別の差を確認できます。攻撃実装は引き続き `attack/M/`・`attack/attack_TS/` に置きます。

[予備戦の匿名化から得た示唆](../prelim/results/20260909/README.md#匿名化から得た示唆)を本戦の比較方針として参照します。有用性の一部を犠牲にするなら、損失に見合う匿名性の改善を同じ条件で確認します。U_specの低さだけを保護の強さとみなさず、希少群へのMIA耐性と総合得点も確認します。

本戦のルール、配布物、実装、提出物を管理します。**[本戦版ルールブック main-process-20260912](rulebook/README.md)** の原本PDFと変更点を2026-09-09に追加しました。

**2026-09-16に本戦・加工フェーズの配布データを配置しました。チームはアウトサイダー / outsider、コホート番号は10（予備戦は24）です。**

**2026-09-27：[攻撃フェーズの作業環境](attack/README.md)を整備しました。** 公式配布物・固定版本戦キット・team10用の実行と検証を用意しています。対象は自己10を除く22コホート。期間は9月28日09:00〜10月11日09:00 JST、採点に通る提出は最大8回です。配布C_10（事務局掲載の提出ID943347）は、最終選択した[M1_v5のC.csv](anonymization/team10/M/M1_v5/C.csv)と完全一致しました。ユーザーの依頼により、攻撃配布原本と参照出力を含む現在の状態を既存の非公開GitHubへ保存します。以下の加工フェーズ記録は各記載日時点の履歴です。

| 内容 | 場所 |
|---|---|
| 本戦の匿名化手法を理解する M1_v5の解説 | [method_explanation/01_anonymization.md](method_explanation/01_anonymization.md) |
| 本戦攻撃データ・実行と検証の手順 | [attack/README.md](attack/README.md) |
| 共通の模擬問題生成・保存・攻撃手法の比較 | [mock_test/README.md](mock_test/README.md) |
| TS本人の本戦攻撃 | [attack/attack_TS/README.md](attack/attack_TS/README.md) |
| 作業者Mの本戦攻撃 | [attack/M/README.md](attack/M/README.md) |
| 本戦加工データの案内・取扱い | [anonymization/team10/README.md](anonymization/team10/README.md) |
| 配布原本 B・B_self・A_bg・schema・utility_ref・README | [anonymization/team10/participant_data/](anonymization/team10/participant_data/) |
| 配布原本6ファイルのサイズ・SHA-256 | [anonymization/team10/data_manifest.json](anonymization/team10/data_manifest.json) |
| 本戦ルールブック | [rulebook/README.md](rulebook/README.md) |

配布README記載の加工フェーズ提出締切は **2026-09-25 09:00 JST** です。2026-09-17に [S1](anonymization/team10/S/S1/README.md) の公式返却値（ユーザー提供）とローカルU=0が一致することを確認しました。修正版 [S2](anonymization/team10/S/S2/README.md) は条件付きroundをsampleに変更し、ローカルU=0.5484134662543412、29テスト・公式形式検証成功、提出ZIP作成済みです。S2のCodaBench提出は未実施です。予備戦の得点に新指標を遡及適用することはありません。

今回のデータ保存はユーザーの明示的依頼による、既存の**非公開**GitHubリポジトリへの保存です。2026-09-17の明示的指定により、認証用 `token.txt` も `anonymization/team10/participant_data/` と `anonymization/team10/模擬提出データ/` の2か所でGit管理します。追加の明示的な許可により、[S1のトークン入り提出ZIP](anonymization/team10/S/S1/submission/team10_S1_defense.zip)も保存します。トークンの内容は文書やログに記載しません。

予備戦の作業は [`../prelim/`](../prelim/README.md) に保存しています。本戦の実装を始めるときは、ルール・採点キット・データの差分を確認してから、必要な実装を明示的に引き継ぎます。予備戦の練習データや得点を、本戦データ・本戦での性能とみなさないでください。

S2完了後の2026-09-17の依頼により、S2の成果物・検証記録・トークン入り提出ZIPも、同じ非公開GitHubリポジトリのmainへの反映対象としました。
