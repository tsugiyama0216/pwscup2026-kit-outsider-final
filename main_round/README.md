# 本戦

[予備戦の匿名化から得た示唆](../prelim/results/20260909/README.md#匿名化から得た示唆)を本戦の比較方針として参照します。有用性の一部を犠牲にするなら、損失に見合う匿名性の改善を同じ条件で確認します。U_specの低さだけを保護の強さとみなさず、希少群へのMIA耐性と総合得点も確認します。

本戦のルール、配布物、実装、提出物を管理します。**[本戦版ルールブック main-process-20260912](rulebook/README.md)** の原本PDFと変更点を2026-09-09に追加しました。

**2026-09-16に本戦・加工フェーズの配布データを配置しました。チームはアウトサイダー / outsider、コホート番号は10（予備戦は24）です。**

| 内容 | 場所 |
|---|---|
| 本戦加工データの案内・取扱い | [anonymization/team10/README.md](anonymization/team10/README.md) |
| 配布原本 B・B_self・A_bg・schema・utility_ref・README | [anonymization/team10/participant_data/](anonymization/team10/participant_data/) |
| 配布原本6ファイルのサイズ・SHA-256 | [anonymization/team10/data_manifest.json](anonymization/team10/data_manifest.json) |
| 本戦ルールブック | [rulebook/README.md](rulebook/README.md) |

配布README記載の加工フェーズ提出締切は **2026-09-25 09:00 JST** です。2026-09-17に [S1](anonymization/team10/S/S1/README.md) で予備戦提出v5の方法を本戦データへ適用し、C.csvと提出ZIPを作成しました。本戦公式キットの形式検証は合格、自己採点U=0（新しい相関指標が0点）です。CodaBenchへの提出は行っていません。予備戦の得点に新指標を遡及適用することはありません。

今回のデータ保存はユーザーの明示的依頼による、既存の**非公開**GitHubリポジトリへの保存です。2026-09-17の明示的指定により、認証用 `token.txt` も `anonymization/team10/participant_data/` と `anonymization/team10/模擬提出データ/` の2か所でGit管理します。トークンの内容は文書やログに記載しません。

予備戦の作業は [`../prelim/`](../prelim/README.md) に保存しています。本戦の実装を始めるときは、ルール・採点キット・データの差分を確認してから、必要な実装を明示的に引き継ぎます。予備戦の練習データや得点を、本戦データ・本戦での性能とみなさないでください。
