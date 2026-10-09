# 杉山採点：本番形式の共同模擬評価

このリポジトリで使う二種類の比較システムは、(1) 本戦形式の固定模擬ケースで順位を比較する [rank_validation](rank_validation/README.md) と、(2) 予備戦の実問題・公開正解で再採点する [mock_TS_using_prelim](mock_TS_using_prelim/README.md) です。用途とデータが異なるため、両者の点数は合算しません。

**本番でのモデル順序との照合と新候補の比較は [rank_validation](rank_validation/README.md) を使います。** 既知4モデルの順位を確認した固定v5の2ケースで、全体MIA＋補正済みAIAの順位比較指標、公式方式の模擬点、希少者MIA上乗せ、ケース・対象別の変動を表示します。Rv2とRv6〜Rv11の一括比較は `rank_validation/run_series.ps1`。未提出モデルの本番順位を保証するものではありません。

**予備戦の実問題・公開正解を使う評価は [mock_TS_using_prelim](mock_TS_using_prelim/README.md) です。** team24の実配布原本、全21チームの匿名化データ・公開回答、運営の正解をコピーし、予備戦ルールでMIA・AIAを採点します。`mock_TS_using_prelim/run.ps1` で公開21回答の公表値との照合、`-Action Run -Plugin <攻撃.py>` で自作攻撃を評価できます。

実行用保存版は、保存先訂正に従い **[attack/retest](../attack/retest/README.md)** に保存しています。`retest/run.ps1` からその保存版を実行できます。

**通常の入口は `sugiyama.ps1` です。** 1つの模擬世界の22対象に対してMIA・AIAを同時に実行し、保存済みの公式 `score_attack` で `score_1`～`score_5` を算出します。既定の攻撃は、M v2のMIAとT v88のAIAを使う、変更していないR v1です。

## 実行

PowerShellでリポジトリ直下から実行します。専用環境がないPCでは、先に `main_round/attack/setup.ps1` を実行してください。

```powershell
Set-Location main_round/mock_test

# 1世界を生成し、R v1を実行して採点。AIA補正値は1,000回で算出
.\sugiyama.ps1 -Output .\local-output\sugiyama\rv1-dev-2026100701

# より難しい加工構成と別の生成元分割で確認
.\sugiyama.ps1 -Profile stress -Split holdout -Seed 2026100702 -Output .\local-output\sugiyama\rv1-holdout-2026100702

# 同じ模擬問題を別の攻撃で再評価
.\sugiyama.ps1 -Action Evaluate -CaseDirectory .\local-output\sugiyama\rv1-dev-2026100701\case -Plugin ..\attack\Attack_R\sugiyama_adapter.py -Output .\local-output\sugiyama\comparison-01
```

`-Action Generate` は問題保存だけを実行します。既存の非空出力先は上書きしません。生成物はGit除外の `local-output/` 配下に限ります。複数モデルを比較する際は、同じ保存問題・同じ補正値のseed・同じ反復数を指定します。既定の補正値seedは保存問題のseedから決まり、攻撃モデルには依存しません。

## 本番との対応

| 項目 | 杉山採点の標準動作 |
|---|---|
| 入力 | B 1,049行、既知ラベル付きB_self、背景A_bg 5,143行、自己AIA質問、22対象のCとAIA質問 |
| 同一世界 | 30コホートの整合した共有関係から22対象を選択。MIAとAIAで同じ人物・加工データを使用 |
| 人数 | 各コホート1,049人・希少24人・共有736人・非共有313人。AIAは各対象326問 |
| 正解 | 模擬人物IDから所属を判定。希少者ラベルはB_selfの既知ラベル由来で、検出器の推定と区別 |
| 攻撃への情報 | 公開テーブルだけ。自己情報から非共有163人を推論し、相手の正解・加工名・生成seedは渡さない |
| MIA | 公式のFPR 1%、全体と希少側の大きい値を対象ごとに採用して22対象平均 |
| AIA | 公式の時間許容幅0.5年、希少側SE補正、対象ごとの補正値φ控除と0での切り上げ |
| 補正値φ | 模擬A_bgから1,049行を非復元抽出し、公式参照AIAを1,000回実行した模擬正解での平均 |
| 集計 | 独立生成の対照対象も含め、22対象を除外せず平均。自己10は対象外 |
| 出力 | score_1=攻撃力合計、2=MIA全体、3=MIA希少、4=AIA全体、5=AIA希少 |

`score_4`・`score_5` は公式同様、φ控除前の内訳です。`score_1` は対象別の最大値・φ控除を経るため、残り4列の単純和や最大値から復元できません。補正値が欠けた場合は、黙って0にせずエラーにします。

加工は丸め、弱・強ノイズ、コピュラ合成、条件付き再抽出、結果列のシャッフル、背景との混合、標的非依存生成を含みます。`mixed` と `stress` の22対象構成は事前固定した検証条件であり、実際の相手の加工方式の推定ではありません。元データのそのままコピーは主評価に含めません。

## 結果の見方

`Run` の出力内で、`case/public/` は攻撃用入力、`case/private/` は採点用正解です。`evaluation/` に以下を保存します。

- `result.json`：公式と同じ5列、MIA/AIA攻撃力、φ控除前の値、入力・実装ハッシュ、再現条件と限界。
- `F_mia.csv`・`F_aia.csv`：その模擬問題の公開入力から再生成した予測。実提出物を模擬正解に流用しません。
- `by_target.csv`・`by_style.csv`：効く対象と効かない対象を確認するための模擬診断。
- `scoring_truth/aia_floor.csv`・`aia_floor_metadata.json`：模擬φ、1,000回のばらつきと平均の標準誤差。

`-FloorRepetitions` を1,000未満にすると動作確認用であることを記録します。正式な模擬比較には既定値を使ってください。複数seedの得点の変動も確認し、候補選択後にholdoutを評価します。

## 再現できる範囲

**採点計算は配布された公式コードと一致しますが、公式点そのものの再現・予測ではありません。** 運営だけが持つ本番の所属・AIA正解・φと、他チームの本当の加工生成過程は入手できません。有限の既知データから新しい模擬人物を生成するため、生成元の再利用・合成分布の違いが残ります。dev/holdoutは生成元行を分けていますが、同じ配布データ由来で、既存R/M/Tにとって完全な未知人物評価ではありません。φの標準誤差は反復抽出の誤差だけを表し、この分布の違いは含みません。公式のR v1得点0.169387840941965に合わせる係数や閾値調整は行っていません。

プラグインは `predict_world(*, candidates, self_labels, background, self_challenge, targets, schema, cfg)` を実装し、`{"F_mia": DataFrame, "F_aia": DataFrame}` を返します。`targets[k]` は `{"release": C, "challenge": QI}` です。引数はコピーされ、正解パスや生成条件を含みません。ただし同じPC上の信頼済みPythonを実行する方式で、悪意あるプラグインのファイルアクセスを隔離するセキュリティ機構ではありません。

旧 `generate.ps1`・`evaluate.ps1`・`local_eval/` は、既存M/T/Rの凍結ハッシュと過去の再現に必要な依存としてのみ保持しています。今後の杉山採点には新しい入口を使います。モデル本体・選択・過去の予測・提出ZIPは変更しません。このツールはトークンを使わず、ZIP作成・GitHub送信・公式提出を行いません。

## 検証

```powershell
..\attack\.venv\Scripts\python.exe -B -m unittest discover -s sugiyama_v2/tests -v
```

公式キットは `main-attack-20260928`、固定コミット `a490490e11a0ec460a5df9fdefb17c72ad459c1e` です。生成器・採点器は攻撃手法から独立し、R v1接続だけを [sugiyama_adapter.py](../attack/Attack_R/sugiyama_adapter.py) に置いています。

2026-10-07の[検証記録](sugiyama_v2/verification_20261007.json)：新42・旧共有29・M v2 24の計95テストが成功。R v1の22対象を2世界で通し、保存予測・正解・φを公式コードへ直接渡した得点と一致しました。

| 模擬条件 | 合計 score_1 | MIA全体 | MIA希少 | AIA全体 | AIA希少 |
|---|---:|---:|---:|---:|---:|
| mixed / dev / seed 2026100701 | 0.678909 | 0.190080 | 0.187500 | 0.483914 | 0.141899 |
| stress / holdout / seed 2026100702 | 0.563669 | 0.129860 | 0.285714 | 0.386800 | 0.082865 |

どちらもφは各対象1,000回で算出しました。**公式のR v1合計0.169387840941965との差は残っています。** 非共有部分の県別分布の偏りも未再現です。これは採点計算の一致を検証した結果であり、公式得点の予測精度が確認できたという意味ではありません。上の2条件は加工構成と生成元分割の両方が異なります。
