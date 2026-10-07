# mock_TS

本選で作成済みの匿名化データCを、手元のデータから作る正解付き問題で攻撃し、`F_mia.csv`・`F_aia.csv`を公式の採点コードで評価します。M1_v5、S1、R系10種類に同じ問題を使い、Cごとに新しいプロセスで攻撃します。

**目的は攻撃手法のローカル比較です。合成対照者や代理ラベルを含むため、本選の公式得点を予測するものではありません。** 問題生成・攻撃・採点を分離し、保存済み問題への再実験と、外部で作った予測CSVの採点に対応しています。

## まず実行する

リポジトリ直下のPowerShellから実行します。既存の `main_round/attack/.venv` を使います。別PCでは先に `main_round/attack/setup.ps1` を実行し、必要な参加者データを配置してください。

```powershell
.\main_round\mock_test\mock_TS\run.ps1
```

出力先を省略すると `main_round/mock_test/local-output/mock_TS/<日時>-run/` に保存します。生成物はGitの対象外です。出力先を指定する場合も、このフォルダーの配下に限定します。中身のある既存フォルダーは上書きしません。

既定では公式スターターの参照攻撃を使い、12種類のC × 3問題＝36回の攻撃・採点と、Cなしの診断3回を行います。終了時に表示する `evaluation/detailed_results.html` をブラウザーで開いてください。

## 固定した模擬条件

| 項目 | 条件 |
| --- | --- |
| C | M1_v5、S1、R1・R3・R4・R5・R6・R7・R9・R10・R11・R12 |
| Rの選択 | R1〜R12から重複なしで10個。選択seed＝20261007 |
| 問題生成 | seed＝2026100701、既定3問題。同じ問題・背景・φを各Cで共有 |
| MIA候補 | 1,049人＝元のBから26人＋A_bgから1,023人 |
| MIA希少候補 | 合計24人。希少所属者が0・1・2人の3条件を順に作る |
| MIA用の背景 | 候補にしたA_bgの1,023人を除外。残り4,120人を渡す |
| AIA問題 | 326人＝Bの既知標的163人＋新規合成対照163人 |
| AIA層別構成 | 標的・対照とも希少13人、発症16人、希少かつ発症4人 |
| AIA補正値 | 各問題で参照AIAを1,000回実行してφ_mockを推定 |
| 標的ID | 常にローカルID `0`。Cを別の標的として同時に渡さない |

`-Seed`、`-Count`、`-SelectionSeed`、`-RCount`で新しい問題を生成できます。Countを増やした場合、希少所属者0・1・2人の条件を繰り返します。平均はこの3条件を等しく扱うシナリオ平均で、公式での出現頻度に基づく期待値ではありません。反復は同じ有限集団の再抽出で、独立した人物集団ではありません。

### MIAの正解

このコンテストのMIAは、候補者が標的Cの元データBに含まれるかを推定する課題です。Cの行との一致判定ではありません。元のBだけでは非所属者を作れないため、手元のA_bgから非所属者を選びます。元のBと完全一致する候補は除外します。

Bの26人は後述のAIA標的163人と重ならないよう選びます。候補を混ぜ直し、新しいrecord_idを付けます。攻撃側の `B.csv` はこの候補表であり、正解の元Bそのものではありません。元のBやB_selfは攻撃に渡しません。

所属者の希少ラベルはB_selfの提供値、非所属者はschemaの公開Mahalanobis検出器による代理ラベルです。両者は同じ定義ではなく、Bでも51人で判定が異なります。希少MIAの値はこの近似に依存します。また希少所属者0人の条件では希少TPRは未定義なので、公式の集約に合わせてscore_3には0を表示し、全体MIAを使います。

### AIAの正解と対照者

手元の `aia_challenge_10.csv` と元のBを7属性で厳密に照合し、B内の標的163人とその真のtimeを特定します。既存challengeの対照者163人はtimeが不明なため、そのまま採点せず、新たな合成対照へ置き換えます。

合成対照は、標的と都道府県・性別・5歳階級・発症・希少層の構成を合わせます。MIAで保留した人物を除いたA_bgの条件付きドナーから測定値・イベント時刻を生成し、B・A_bg・生成済み対照との一致や極端な近似コピーを避けます。疎な層ではドナー検索条件を段階的に緩め、その履歴をprivate側へ保存します。標的本人の測定値やtimeはコピーしません。

**合成timeと合成希少層が、今回の模擬上の正解です。** 未発症・未死亡なら10年で打切り、イベント時刻はドナー時刻に小さな揺らぎを加えます。測定値とtimeは別のドナーから抽出するため、両者の関連は完全には保存されません。また近似コピーを避ける制約も対照者の分布を変えます。こうした差が攻撃の手がかりになる可能性は残り、φの再推定でも手法ごとの生成由来の偏りをすべて補正できるわけではありません。

## 自作の攻撃で同じ問題を評価する

攻撃ファイルに次の関数を用意します。ひな型は [example_attack.py](example_attack.py) です。攻撃の実装は `main_round/attack/<担当>/` に置き、評価器とは分けてください。

```python
def predict(*, c, background, candidates, challenge, cfg):
    # この呼出しに渡された公開入力だけで学習・推論する。
    # mia: candidatesの入力順、0〜1の所属確信度
    # aia: challengeの入力順、0以上で有限のtime推定値
    return {"mia": mia, "aia": aia}
```

入力DataFrameにはB_selfのラベル・生成seed・元の行番号・Cの版名を含めません。代わりに `{"F_mia": dataframe, "F_aia": dataframe}` を返すこともできます。既存の22標的一括 `predict_world()` をこのまま渡すことはできません。1標的の公開入力を受けるアダプターが必要です。

最初の実行で作ったsuiteを使い続ける例です。以下の `reference_20261007_v1` は検証したこのPCの保存名です。生成したsuiteやHTML結果はGitHubには含めません。別PCでは、先に次のコマンドで同じ条件の問題・参照結果を生成するか、既に生成したsuiteのパスへ読み替えてください。

```powershell
.\main_round\mock_test\mock_TS\run.ps1 `
  -Output .\main_round\mock_test\local-output\mock_TS\reference_20261007_v1
```

同名の出力先が存在する場合は上書きせず、保存済みsuiteをそのまま利用します。

```powershell
$mockSuite = '.\main_round\mock_test\local-output\mock_TS\reference_20261007_v1\suite'
.\main_round\mock_test\mock_TS\run.ps1 -Action Benchmark `
  -SuiteDirectory $mockSuite `
  -Plugin .\main_round\mock_test\mock_TS\example_attack.py
```

`-Models M1_v5,S1`、`-Cases case_001`で対象を絞れます。動作確認だけなら `-FloorRepetitions 20`も使えますが、レポートに診断用と表示し、本比較では1,000回に戻します。

### 入力を渡し、予測CSVを後から採点する

```powershell
$mockSuite = '.\main_round\mock_test\local-output\mock_TS\reference_20261007_v1\suite'
$mockBase = '.\main_round\mock_test\local-output\mock_TS\my_experiment'

# 1. case_001 / M1_v5だけの公開入力を出す
.\main_round\mock_test\mock_TS\run.ps1 -Action Export `
  -SuiteDirectory $mockSuite -CaseId case_001 -Model M1_v5 `
  -Output "$mockBase\input"

# 2. 自作攻撃を新しいプロセスで実行
.\main_round\mock_test\mock_TS\run.ps1 -Action Attack `
  -BundleDirectory "$mockBase\input" `
  -Plugin .\main_round\mock_test\mock_TS\example_attack.py `
  -Output "$mockBase\predictions"

# 3. 保存したF_miaとF_aiaを採点
.\main_round\mock_test\mock_TS\run.ps1 -Action Score `
  -SuiteDirectory $mockSuite -CaseId case_001 -Model M1_v5 `
  -PredictionDirectory "$mockBase\predictions" -Output "$mockBase\evaluation"
```

2の代わりに別プログラムで予測CSVを作っても採点できます。必要な形式は次のとおりです。IDは配布入力の値を使い、行の並び替えは可能ですが、欠損・重複・余分なIDは拒否します。

| ファイル | 列 | 行数 |
| --- | --- | --- |
| F_mia.csv | `record_id,0` | 1,049 |
| F_aia.csv | `target_k,challenge_row_id,time_hat` | 326、target_kは全て0 |

Runner経由の予測には入力と予測のハッシュを添付し、別問題・別Cの取り違えを拒否します。外部CSVだけの場合は対応する問題・Cの指定を利用者の申告として扱い、その限界を記録します。

## 得点とレポート

固定した公式 `pwscup2026.kit.codabench.scoring_io.score_attack` を呼びます。自己標的の除外を避けるため、ローカル標的0を `team_id=None` で採点します。

| 表示 | 意味 |
| --- | --- |
| score_1 | MIA力＋φ_mock控除後のAIA力 |
| score_2 / score_3 | 全体／希少MIAのTPR@FPR 1% |
| score_4 / score_5 | 全体／希少AIAの正規化超過成功率。φ控除前 |
| MIA力 | 全体・希少MIAの最大値 |
| AIA力 | 全体・希少AIAの最大値からφ_mockを引いて0〜1に制限 |
| score_6 | 未計算。公式最終順位や22標的の総合評価を模倣しない |

AIAのtime成功判定は±0.5年、希少AIAには公式の1.64×SE補正を使います。MIAの匿名化評価用置換帰無補正とは別の、攻撃側の採点です。

φ_mockは、許可背景から1,049人を重複なしで抽出する参照AIAを繰り返し、その公式AIA値を平均します。標準偏差・標準誤差・seed・入力ハッシュを保存します。提供済みφ₁₀＝0.026887は元の公式問題の値なので、問題を変更した本mockには流用せず、感度確認として別掲します。

- `detailed_results.html`：C・問題別の得点、問題平均、内訳、人数、φ推定、解釈上の限界。
- `summary.csv`：表形式の得点。
- `results.json`：公式採点結果と実行記録。
- 各Cの `predictions/`：F_mia・F_aia、予測ハッシュ、攻撃ログ。
- `suite/<case>/private/`：模擬正解、対照者、生成元、条件。攻撃へ渡さない。

各問題でCを使わない背景密度のMIAと背景のみのAIAも計算します。これは候補の出所だけで解けてしまわないかを見る診断です。Cありの得点から機械的に差し引く補正はしません。

## 分離の範囲と残る限界

各CをUUID名の新しい作業フォルダーとPythonプロセスへ渡し、他のC・その予測・前のプロセス内の学習状態を渡しません。ただし、これは信頼する攻撃コードを対象とした実行分離で、ローカルファイルの読み取りを禁止するOSサンドボックスではありません。プラグインから元のB・A_bg全体・private正解・別Cの予測を読まないでください。

MIA候補として保留したA_bgを、既存学習済みモデルやキャッシュから取り除くことはできません。各実験は渡された背景から学習し直す必要があります。さらに既存Cの作成時にはA_bgを利用したものがあるため、完全に未使用の母集団による評価とは呼べません。合成対照、代理希少ラベル、少数のMIA陽性、既存Cの生成履歴が主な近似です。

## 検証と参照

```powershell
Push-Location .\main_round\mock_test
try {
  ..\attack\.venv\Scripts\python.exe -B -m unittest discover -s mock_TS/tests -v
} finally { Pop-Location }
```

人数・希少構成・背景除外・対照者の衝突・再現性、公式スコアとの一致、φ推定と公式参照予測の一致、CSVの形式拒否、プラグイン実行分離を検証します。確認した実行は [verification_20261007.json](verification_20261007.json) に記録します。

仕様の参照元は本選ルールブック§4.2、§6.2.2、§6.3と手元の公式スターターです。RinkaAbeさんが2026-10-07に追加した評価器（[9ef35df6](https://github.com/tsugiyama0216/pwscup2026-kit-private/commit/9ef35df6a1410d3997d15cddf7fed66a3670758e)、保存版 [attack/retest](../../attack/retest/README.md)）の公式採点・φ推定の扱いも参照しました。mock_TSは既存Cを別々に比較する目的で追加した独立ツールです。既存評価器・攻撃・Cの保存済み内容は変更しません。
