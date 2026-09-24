# S_DPCopula_ep15

2026-09-24のユーザー指定に基づく、単独ε=15・δ=0の比較用モデル。ε20の固定設定を共有し、予算だけを3/4にした。ヒストグラム3、Cox補正2.4・3.6・6。1回の新規計測と固定後処理で作成し、採点・攻撃結果に合わせた再生成はしていない。

ローカル有用性Uは **0.805932959820**。全内訳は [非DP診断](private_diagnostics/utility.json)、DOMIAS・AIAとの比較は [共通比較](../../M/evaluation/domias_main_round_20260924/README.md) を参照。

`C.csv`、`model.json`、`manifest.json` は [元の実行場所](../S_Copula/epsilon15/release/) とバイト一致。[ORIGIN.json](ORIGIN.json) に照合記録を保存した。共通の実行入口は `../S_Copula/epsilon15/release15.py`、固定設定は `config15.json`、別置き耐久台帳は `release_ledger/reservation.json`。再実行が保存モデルを返し、Bを読まないことを確認した。

保証の条件は [設計文書](../S_Copula/epsilon15/PRIVACY_DESIGN_15.md) を参照。同梱の設計文書コピー内の相対パスは元の実装場所が基準。ε20との基本合成上限は35、ε1・3・10・15・20の5出力では49。削除済みε400を含むDPCopula計測の使用履歴は449。非DPの採点や他モデル・資料全体への保証ではない。

このPCへの保存のみ。GitHubアップロード・CodaBench提出は未実施。
