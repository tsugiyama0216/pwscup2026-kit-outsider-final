# S1：予備戦提出v5方式を本戦team10へ適用

2026-09-17作成。作業者Sの最初の匿名化モデルです。予備戦で実際に提出したv5の手法を、本戦の配布データで1回実行しました。

**形式検証済みの提出ZIPを用意しましたが、本戦の有用性自己採点は U=0 です。今回追加された相関指標が0点になるため、そのままの本提出は推奨しません。** 予備戦方式の比較基準として保持します。CodaBenchへの提出は行っていません。

## 成果物

| ファイル | 内容 |
|---|---|
| [C.csv](C.csv) | 本戦team10の匿名化データ、1,049行×13列、UTF-8 |
| [submission/team10_S1_defense.zip](submission/team10_S1_defense.zip) | ZIP直下にC.csvと本戦team10のtoken.txtのみ。利用者がアップロードする提出ファイル |
| [validation.json](validation.json) | 配布原本・生成履歴・形式・ZIP内容の検査結果 |
| [score.json](score.json) | 本戦公式コードによるローカル有用性と内訳 |
| [evaluation_provenance.json](evaluation_provenance.json) | 採点器、入力、出力、ライブラリの来歴 |
| [config/aim_main_team10_S1.json](config/aim_main_team10_S1.json) | 本戦の公開A_bg/schemaから構成したモデル設定 |
| [privacy_report.json](privacy_report.json) / [release_ledger.json](release_ledger.json) | 1回の生成記録・ハッシュ・設定上のρ=0.1 |
| [source_provenance.json](source_provenance.json) | 予備戦の実提出コード・設定との対応 |

ZIPは秘密トークンを含むためGit管理対象外です。トークンは共通配布場所から直接ZIPへ格納し、モデルフォルダー内に単独のtoken.txtは作成しません。

## 引き継いだ手法

1. 本戦の公開A_bgから数値の分位ビン、カテゴリの代表値、発症・死亡のCox係数を構成。
2. 本戦Bを離散化し、OpenDPの一変量測定・固定9組・AIMの候補39組で1,049行を合成。
3. 公開A_bgを用いて数値へ復号し、公開の希少判定器と希少率で7変数の組を調整。
4. 公開A_bgのCoxリスクが最も近い行からtime/onset/deathを一組で割り当て。

ρ=0.1、oneway_split=0.35、fixed:adaptive=3:1、measure_split=0.9、rounds=16、max_size_mb=32、希少ドナーseed=2・距離上限5.5、死亡リスク重み0.31を予備戦v5から維持しています。数値ビンとCox係数は本戦A_bgだけで作り直しました。B_selfの真の希少ラベルは生成に使わず、自己採点時だけ参照しています。

予備戦v5は親Cの転帰を再調整していましたが、その親Cの転帰は最終ドナー選択に使われません。そのため、今回は同じv5設定を最初から適用しています。予備戦Cの使い回しではありません。

## 本戦の自己採点

公式 [main-process-20260912](https://github.com/pwscup/pwscup2026-kit/releases/tag/main-process-20260912)、コミット `3dd71bd3436f2abfdbb64da97e860f22729ea054` のコードを利用しました。採点依存パッケージは公式pyprojectの固定バージョンと一致しています。

| 指標 | 得点 |
|---|---:|
| U_gen | 0.000000 |
| U_spec | 0.633593 |
| U_rare | 0.835809 |
| U_valid | 0.882491 |
| **有用性U（4観点の最小）** | **0.000000** |

U_genの相関指標では、36組のSpearman相関の平均距離が0.489979となり、尺度0.4を超え、相関得点が0に切り詰められました。周辺分布平均0.940515、pMSE得点0.984902、time_dz=1、裾の得点0.998871に対して、相関の保存が今回の弱点です。希少検出は66件でゲート8件以上、完全重複行は0件です。

これはローカル有用性の結果です。公式提出結果、匿名性、総合得点ではありません。得点を見て設定変更や再生成は行っていません。

## 再検証・実行

このPCでは既存のOpenDP環境と公式採点用Python環境を再利用します。`run.ps1` の `-AimPython` / `-ScorePython` / `-KitRoot` で別環境を指定できます。生成環境の依存関係は同梱のpyproject.tomlとuv.lock、採点環境は公式キットのstarter/pyproject.tomlを参照してください。

```powershell
# S1フォルダーで実行。保存済みCの検査と再梱包だけを行う。
.\run.ps1 -Mode PackageExisting
# 保存済みCを本戦キットで自己採点。
.\run.ps1 -Mode Score
# 入力・公開設定の検査のみ。新しいDP生成は行わない。
.\run.ps1 -Mode Preflight
```

公式キットは `local-output/official-kit/` に固定コミット版を保存しています。このローカルフォルダーはGit対象外です。別PCでは公式の[固定コミットZIP](https://codeload.github.com/pwscup/pwscup2026-kit/zip/3dd71bd3436f2abfdbb64da97e860f22729ea054)を `main-process-20260912.zip` の名前で保存し、同じ場所に展開します。展開先のリポジトリルートを `-KitRoot` で指定してください。検証時はZIPのSHA-256と、展開した採点コードのバイト一致も確認します。

`Generate` は未生成モデル用の実行モードですが、S1は生成済みなので既存の台帳・出力が再実行を拒否します。追加生成は別モデル番号で管理してください。DP測定と復号には乱数があるため、新規実行で同一CSVにはなりません。提出物の再現には保存済みCとハッシュを使います。

元の生成器・公開設定生成器は予備戦の実ファイルからバイトを変えずコピーしました。設定内のbuilder文字列が旧配置を示すのはそのためで、今回の実行ファイルは `scripts/build_public_config.py` です。秘密トークンの値は記録に含めていません。
