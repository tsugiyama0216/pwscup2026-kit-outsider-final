# Sのモデル

作業者Sが作成した本戦匿名化モデルを格納します。

- モデルごとにフォルダーを分けます。DPCopulaは指定名称 `S_DPCopula_ep1/`、`S_DPCopula_ep3/`、`S_DPCopula_ep10/`、`S_DPCopula_ep15/`、`S_DPCopula_ep20/` で整理しています。
- 各モデルの実装・設定・評価結果を、そのモデルのフォルダーにまとめます。
- 共通の配布データは [`../participant_data/`](../participant_data/) を参照します。

| モデル | 方式 | 状態 |
|---|---|---|
| [S1](S1/README.md) | 予備戦提出v5（OpenDP AIM＋公開A_bg後処理）を本戦データへ適用 | U=0。ユーザー提供の公式結果とローカル採点が一致。元の成果物を保存 |
| [S2](S2/README.md) | S1の条件付きround不具合をsampleで回避、DPモデル保存を追加 | C.csv・提出ZIP作成、公式形式検証済み。ローカルU=0.548413。CodaBench未提出 |
| [S_DPCopula_ep1](S_DPCopula_ep1/README.md) | 公開相関Gaussian copula＋DP補正、単独ε1・δ0 | 保存済み出力、ローカルU=0.612881。未提出 |
| [S_DPCopula_ep3](S_DPCopula_ep3/README.md) | 公開相関Gaussian copula＋DP補正、単独ε3・δ0 | 保存済み出力、ローカルU=0.710956。未提出 |
| [S_DPCopula_ep10](S_DPCopula_ep10/README.md) | 公開相関Gaussian copula＋DP補正、単独ε10・δ0 | 保存済み出力、ローカルU=0.759902。未提出 |
| [S_DPCopula_ep15](S_DPCopula_ep15/README.md) | 公開相関Gaussian copula＋DP補正、単独ε15・δ0 | 2026-09-24に1回作成、ローカルU=0.805933。未提出 |
| [S_DPCopula_ep20](S_DPCopula_ep20/README.md) | 公開相関Gaussian copula＋DP補正、単独ε20・δ0 | 保存済み出力、ローカルU=0.923104。未提出 |
| [S_Copula](S_Copula/README.md) | 共通実装・固定設定・予算台帳・元の実行履歴 | ε別の整理済みモデルは上記5フォルダー。ε400旧モデル・CSVは削除済み |

S1の0点検証は [検証記録](research/s1_official_verification_20260917/README.md)、修正の効果は [S2の同じモデルによる比較](S2/README.md#同じモデルによる不具合の確認) を参照してください。
