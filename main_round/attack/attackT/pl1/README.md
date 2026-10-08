# pl1：杉山さんの予備選攻撃を本戦へ移植

予備選の最終v9で使われたMIA v4と、AIAの予備選v5合議方式を、本戦team10・22対象へ接続した版です。保存先はユーザー指定の `main_round/attack/attackT/pl1/`。既存の `Attak_T` やM/Rのモデルは変更しません。

MIAは距離・輸送割当と既知非共有者によるconformal順位の融合です。対象間共有25～27人、対象全体の共有人数は `round(736*22/29)` という明示的な事前仮定を使用します。AIAは `consensus_v4_copula__v4_event_margin` を全対象へ適用します。予備選v9の対象番号別の過去最高CSVを本戦へコピーすることはありません。予備選の新データでv9を完全再現する版でもありません。

`protocol.json` に採点前の方式固定と検証条件を記録しました。予備選の公開正解、本戦の他者正解、加工方式ラベルを攻撃へ渡しません。既存の `Attak_T/prelim_transfer/methods.py` を通じて、保存済み予備選コードの数学的処理を使用します。

## 提出データの再生成

リポジトリ直下のPowerShellで、既存本戦環境を使用します。新しいPCでは先に `main_round/attack/setup.ps1` を実行してください。

```powershell
.\main_round\attack\attackT\pl1\run.ps1 -OutputDirectory .\main_round\attack\attackT\pl1\regenerated
```

出力先はpl1内の未使用ディレクトリに限ります。2つの独立プロセスでCSVを再生成し、バイト一致を確認して公式形式・実際のteam10 IDを検査します。ZIP直下は `F_mia.csv`、`F_aia.csv`、配布原本の `token.txt` の3件です。トークン入りZIPはPCだけに保存しGitから除外します。CodaBench提出は行いません。

## 最新版の杉山自己採点

本戦形式の22対象共同評価 `main_round/mock_test/sugiyama_v2/` を使用します。保存版は `main_round/attack/retest/`。pl1専用の入口を付けました。

```powershell
.\main_round\attack\attackT\pl1\score.ps1 -Split dev
.\main_round\attack\attackT\pl1\score.ps1 -Split holdout
```

公式の固定採点実装を使用し、AIA補正値φは対象ごとに1,000回の模擬抽出で推定します。加工構成mixed/devとstress/holdoutは異なるため、単純な人物分割だけの比較ではありません。既使用の生成元データを再利用する模擬試験です。**公式点の予測ではなく、本戦pl1の公式得点は未測定です。**

採点結果と検証記録は `evaluation/` に保存します。大きな生成問題・採点用正解は共通のignored `main_round/mock_test/local-output/sugiyama/` に残します。

## 保存先

- GitHub: https://github.com/tsugiyama0216/pwscup2026-kit-private/tree/main/main_round/attack/attackT/pl1
- PC: このREADMEと同じフォルダー。提出物は `submission/`。
- ZIP: `submission/team10_pl1_attack.zip`（PCのみ）。
