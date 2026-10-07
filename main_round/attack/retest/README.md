# retest

2026-10-07に改善した杉山採点の実行用保存版です。ユーザーの保存先訂正に従い、既存の `main_round/attack/retest/` に保存しています。

22対象のMIA・AIAを同じ模擬世界で評価し、配布された公式コードで集計します。AIA補正値φは既定1,000回の参照攻撃で算出します。既定モデルは変更していないR v1（M v2 MIA＋T v88 AIA）です。

## 実行

PowerShellでリポジトリ直下から実行します。

```powershell
# 模擬問題を生成してR v1を採点
.\main_round\attack\retest\run.ps1

# 難しい加工構成と別の生成元分割
.\main_round\attack\retest\run.ps1 -Profile stress -Split holdout -Seed 2026100702

# 問題生成だけ
.\main_round\attack\retest\run.ps1 -Action Generate
```

保存問題を再評価するときは `-Action Evaluate -CaseDirectory <問題フォルダー>`、別の攻撃は `-Plugin <Pythonファイル>` を指定します。`-Output` を省略すると、既存の共通出力領域 `main_round/mock_test/local-output/retest/<日時>-<処理名>/` に保存します。出力はGit除外で、既存の非空フォルダーは上書きしません。

## 保存内容

- `run.ps1`：この保存版を実行する入口。
- `sugiyama_v2/`：問題生成・読込・採点・実行コードと42テスト。内部のPythonパッケージ名は維持しています。
- `sugiyama_v2/verification_20261007.json`：元の95テストとR v1の2通し評価の記録。元のバイト・当時のパスを保持しています。
- `save_verification_20261007.json`：現在の配置と、移動後の実行確認の記録。
- `save_verification_before_attack_move_20261007.json`：保存先訂正前の記録。当時のパスを保持しています。

入力と公式キット・専用Python・凍結モデルは、同じリポジトリ内の既存配置を参照します。独立したリポジトリへこのフォルダーだけをコピーして動かす構成ではありません。入力位置と初期化の変更だけで、採点式・生成方法・モデルは元の杉山採点と同じです。詳しい評価仕様は[杉山採点の説明](../../mock_test/README.md)を参照してください。

公式コードと同じ計算でも、非公開の本番正解・φ・実際の加工分布は再現できません。元のR v1模擬合計はmixed/devで0.678909、stress/holdoutで0.563669であり、公式の0.169388との差が残っています。公式点の予測値とは扱いません。トークン・提出ZIP・仮想環境はこの保存版にコピーしていません。

## テスト

```powershell
Set-Location main_round/attack/retest
..\.venv\Scripts\python.exe -B -m unittest discover -s sugiyama_v2/tests -v
```
