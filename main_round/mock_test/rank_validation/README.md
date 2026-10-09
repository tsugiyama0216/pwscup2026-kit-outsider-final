# 本番順位との整合性を検査する自己評価

保存済み予測の公式再採点、本番と同じ実行経路での共同評価、公開分布に合わせた新しい確認問題、候補モデルのペア比較を行うツールです。既存の公式採点式・攻撃モデル・予測・過去の生成器は変更せず、新しい生成器と比較指標を別に追加しています。

数値そのものの予測より、画像で確認された **Rv2 > Rv1 > pl1 > Tv105** の大小関係を検証します。既知4モデルの一致と、未提出モデルの順位予測精度は別に扱います。新候補の入口は末尾の `run_candidate.ps1`、固定指標の確認レポートは `stable_report.py` です。

## 最終確認の結果（2026-10-09）

未使用だったv5確認2ケースで、**それぞれ本番の全6組の大小関係が一致**しました。2ケースの等重み平均も一致し、同じ対象を両ケースから1つずつ外す22通りでも全順位を維持しています。新指標の値は予測本番総合点ではありません。

| モデル | 本番総合の表示値 | 順位比較指標・2ケース平均 |
|---|---:|---:|
| Rv2 | 0.1967 | 0.886144 |
| Rv1 | 0.1694 | 0.823943 |
| pl1 | 0.0996 | 0.797523 |
| Tv105 | 0.0981 | 0.669780 |

同じv5ケースに公式式をそのまま使う場合は、平均も各ケースも5/6一致で、Rv1とpl1が逆転します。新指標でも単独ケースの対象除外は43/44通りが全順位一致で、seed61から対象13を除いた場合だけRv1とpl1が逆転しました。差が小さい候補はケース別・対象別の変動も見て判断してください。

詳細：[保存済み最終レポート](evidence/publication/v5-final/report.html)、[全検証結果](evidence/publication/v5-final/result.json)。初回175テスト成功。新候補用コマンドも定数モデルで予測・採点・比較まで完走し、同点MIAの期待値0.01と基準モデルを下回る判定を確認しました。

この成功は**既知4モデルの順位確認**です。公開特徴が十分に合わない対象が11/22あり、母集団も共通のため、未知の候補の本番順位まで検証できたわけではありません。初回検証はGitHub保存前の記録です。今回のアップロード依頼によりコード・選択した検証記録を既存private mainへ保存します。公式提出や試行消費は行いません。

## GitHubから取得したPCでの実行

生成ケース・予測・補正値の作業フォルダはGit除外です。コード取得直後に新候補を実行する場合は、先に固定2ケースと4基準モデルを再生成してください。既存の本戦専用Pythonを使い、新規PCだけ `main_round/attack/setup.ps1` で環境を準備します。以下はリポジトリ直下から実行します。

```powershell
# 新規PCのみ
.\main_round\attack\setup.ps1
.\main_round\attack\prepare_attack_m.ps1

# 固定2ケースを生成・採点。既に完了した保存先は上書きしません。
$rvOutput = 'main_round/mock_test/local-output/ranking-v5-20261009'
foreach ($rvSeed in @(2026100961,2026100962)) {
    $rvCase = "$rvOutput/cases/confirm-$rvSeed"
    $rvPanel = "$rvOutput/panels/confirm-$rvSeed"
    .\main_round\mock_test\rank_validation\run_joint.ps1 -Action Generate -Generation v5 -Seed $rvSeed -Output $rvCase
    .\main_round\mock_test\rank_validation\run_joint.ps1 -Action Prepare -Case $rvCase -Output $rvPanel
    .\main_round\mock_test\rank_validation\run_joint.ps1 -Action Batch -Panel $rvPanel
    .\main_round\mock_test\rank_validation\run_joint.ps1 -Action Predict -Panel $rvPanel -Model pl1
}
.\main_round\mock_test\rank_validation\run_joint.ps1 -Action Report -Generation v5 -Panels @("$rvOutput/panels/confirm-2026100961","$rvOutput/panels/confirm-2026100962") -Output "$rvOutput/new-reference-report"

# Rv6〜Rv11を同じ2ケースで一括採点し、Rv2と比較
.\main_round\mock_test\rank_validation\run_series.ps1
# 新しい任意候補
.\main_round\mock_test\rank_validation\run_candidate.ps1 -Plugin <model_plugin.py> -Name candidate_v1
```

Rv6〜Rv11は `main_round/anonymization/team10/attack/attackR/` の固定済み攻撃モデルです。`series_worker.py` が同じケースのRv2予測を入力ハッシュ・モデル署名・予測バイトで照合して共用し、公開入力だけを `rv6_series.model.predict_all` に渡します。本番保存CSVを模擬ケースに流用しません。`run_series.ps1 -ReportOnly` なら保存済み7モデルから新しい出力先に比較レポートだけを作れます。

## Rv2・Rv6〜Rv11の採点結果

同じv5確認2ケースで6候補を初めて比較しました。**Rv8だけがRv2を上回りました。** Rv8の改善は全体MIAで、AIAはRv2と同じです。モデルや設定は変更していません。

| 自己採点順位 | モデル | 順位比較指標 | Rv2との差 | 公式方式の模擬合計 |
|---:|---|---:|---:|---:|
| 1 | Rv8 | 0.904818 | +0.018674 | 0.990482 |
| 2 | Rv2 | 0.886144 | 0 | 0.979675 |
| 3 | Rv10 | 0.866382 | −0.019762 | 0.959913 |
| 4 | Rv6 | 0.860022 | −0.026121 | 0.936071 |
| 5 | Rv11 | 0.853181 | −0.032963 | 0.927481 |
| 6 | Rv7 | 0.846526 | −0.039618 | 0.925197 |
| 7 | Rv9 | 0.830757 | −0.055387 | 0.924288 |

Rv8のRv2との差は各ケースで +0.019586 / +0.017762。2ケースから同じ対象を1つずつ外す22通りでは +0.014948〜+0.023373、各ケース単独の44通りでも改善を維持しました。公式方式の模擬合計でも同じ7モデル順です。これは模擬問題での比較で、本番総合点や本番順位の予測を保証しません。6候補の選択に使ったため、選ばれたRv8への独立確認としても扱いません。

[比較レポート](evidence/rv6_series_20261009/report.html)／[順位CSV](evidence/rv6_series_20261009/ranking.csv)／[全記録](evidence/rv6_series_20261009/result.json)。ケース別成分、全21ペアの差、希少MIA上乗せ、対象除外の変動を含みます。

本番公開入力での再推論は12CSV中11ファイルがバイト一致。Rv8 MIAだけ2値に最大1.39×10⁻¹⁷の丸め誤差があり、数値完全一致とは扱いません。ただし全6モデル×22対象のMIA順位（同点を含む）は完全一致し、AIA全6CSVはバイト一致しました。したがって元のMIA順位型採点に影響する差はありません。モデル・CSVの丸め直しはしていません。[再現検証](evidence/rv6_series_20261009/verified_rank_parity.json)を参照してください。

## Rv3・Mv1・Mv2の追加採点（2026-10-09）

前回と同じ固定v5の2ケース・22対象・共通補正値で、モデルを変更せず追加評価しました。Rv3の正本は `main_round/anonymization/team10/atack/Rv3/`、Mの正本は `main_round/attack/Attack_M/v1/`・`v2/` です。

| 全10モデル中の順位 | モデル | 順位比較指標 | Rv2との差 | 公式方式の模擬合計 |
|---:|---|---:|---:|---:|
| 8 | Mv2 | 0.818109 | -0.068034 | 0.870557 |
| 9 | Mv1 | 0.728281 | -0.157863 | 0.805204 |
| 10 | Rv3 | 0.064477 | -0.821667 | 0.109931 |

Mv2はMv1より +0.089829 ですが、追加3モデルはいずれも両ケースでRv2を下回りました。前回7モデルの指標・順位は不変で、1位Rv8・2位Rv2です。本番公開入力で3モデルを再生成し、保存済み6CSVすべてがバイト一致しました。実本番の得点や順位が判明したわけではありません。266テスト合格、固定モデル・保存予測・過去記録は保持しています。

[全10モデル比較レポート](evidence/additional_20261009/report.html)／[順位CSV](evidence/additional_20261009/ranking.csv)／[検証記録](evidence/additional_20261009/verification.json)。既存v5パネル・前回7モデルが揃っていれば、リポジトリ直下から `main_round/mock_test/rank_validation/run_additional.ps1` で追加3モデルを実行します。採点済みなら `-ReportOnly` で新しい比較レポートを作れます。既存予測・結果は上書きしません。

## 公開前の再現確認

公開予定ソースを別の場所へ書き出し、生成済みケース・予測・作業キャッシュなしで225テスト（216＋9）を通過しました。同じ固定Python環境でseed 2026100961を再生成し、公開入力と非公開正解の97ファイル、およびmanifestがすべてバイト一致しました。公式原本94ファイルのハッシュも一致し、実行用ソース55ファイルは検証した書き出し版と同一です。これは同じPC・既存環境での配置独立性の確認で、新規PCでの実測ではありません。[今回の検証記録](evidence/publication_current_20261009/verification.json)に環境とテストログを保存しています。

## 提供ZIPに対する歴史監査（任意）

GitHub `9e5e2af75d38709ecf9b66785e51e79f2d087b94` を取得した作業コピーで実装しました。元の作業フォルダーの未コミット変更は保持しています。PowerShellでこの作業コピーのリポジトリ直下から実行します。

```powershell
.\main_round\mock_test\rank_validation\run.ps1 `
  -Handoff '<score_handoff_20261009を展開したフォルダ>'
```

既存の本戦Pythonを使います。worktreeに環境がなければ元のcheckoutの同環境を読み取ります。自動検出できない場合は `-Python <python.exe>` を指定してください。出力は `main_round/mock_test/local-output/rank-validation/<日時>/`。既存の出力は上書きしません。

`report.html` が閲覧用、`comparison.json` が全比較、`official_rank_checks.csv` が公式との大小比較、`model_summaries.csv` が実験別集計です。`public-profiles/` には本戦22対象と手元12加工の公開特徴・ハッシュ・適用範囲の検査を保存します。

画像からの追加公式値は [evidence/official_totals_20261009.json](evidence/official_totals_20261009.json) を別資料として読み込みます。既知の厳密値は保持し、画像だけのpl1/Tv105は4桁表示値と丸め範囲を記録します。モデル名未確認の内訳JSONは未確認のままです。

## 最初の保存済み評価に対する監査結果

画像で確認した公式順位は **Rv2 (.1967) > Rv1 (.1694) > pl1 (.0996) > Tv105 (.0981)**。前2モデルの厳密値はZIPから確認しました。画像の提出IDは順に966378、954439、967868、968207です。

| 同条件比較（B−A） | 模擬合計差 | 本番合計差 | 大小 |
|---|---:|---:|---|
| Rv1→Rv2／共同22対象・確認 | +.068634 | +.027273 | 同方向 |
| Rv1→pl1／共同22対象・holdout | −.085747 | 約−.069788 | 同方向 |
| Rv2→Tv105／mock_TS test1 | +.077991 | 約−.098561 | 逆転 |
| Rv2→Tv105／mock_TS test2調整 | +.016026 | 約−.098561 | 逆転 |
| Rv2→Tv105／mock_TS test2確認 | +.021368 | 約−.098561 | 逆転 |

総合点が4モデルあっても、同じ模擬条件で照合できるモデル組は3ペアです。同じ公式ペアを複数試験で比較した回数を独立した検証数にしません。pl1の全体MIAには共同22対象でも本番との成分順位逆転があります。

### 計算ではなく、評価用データが主な問題

- ZIP 2,761ファイルを照合し、846再採点すべて保存値と一致。最大絶対差 `9.8879238130678e-17`。
- 予備戦公開21チームの5,589数値を再現。MIA420組・AIA419組、最大差 `5.551115123125783e-17`。欠落AIA1組は公式の既知欠落として扱います。
- mock_TSの12加工は同じteam10のB由来で、そのうち10加工がR系列。AIA標的163人は固定、対照は合成。MIA希少ラベルも陽性と陰性で由来が異なります。
- 公開QI距離では本戦15/22対象、対背景距離比では16/22対象が手元12加工の範囲外。どちらかで範囲外になる対象は17/22です。これは匿名化強度の得点ではなく、評価データの特徴の範囲不足です。
- Tv105は、希少所属0人の層で3試験すべてRv2より低下しました。平均改善は少数の希少所属がある条件に強く依存します。

## 判定方法

公式コードの得点はそのまま保存し、補正係数の学習や4つの内訳からの総合点復元は行いません。

1. 同じ実験・問題・加工・正解・設定・採点器だけをペア比較します。片側欠落・ID違反・重複・非有限値・公式値の矛盾は拒否します。
2. 問題別、加工別、問題／加工を一つ除いた結果を表示します。これらの範囲は信頼区間ではありません。
3. 12加工のM/S/R系列を等重みにする別シナリオと、希少所属0/1/2人別の差を表示します。系列等重みは本戦の構成比推定ではありません。希少所属0人の希少TPRは評価不能です。
4. 既知の本番総合順位を逆転する試験、主要な層で勝敗が反転する候補、公開特徴の範囲が不足するmock_TS比較は採用保留にします。一つの悪いケースだけで永久不採用にはしません。
5. 既存条件で一貫していても、未知モデルの本番順位が確認できたとは表示しません。

このゲートは今回の資料を見た後に作った**回顧的な診断**です。Tv105の誤選択を検出できたことは、未提出の候補に対する予測精度の検証にはなりません。

## 新しいモデルへの利用

新候補は、先にモデルを固定し、比較相手と同じ保存問題に実行してください。別々のseed・加工集合で得た平均値はこのツールで結合しません。出力を同じ正規化レコードにすれば `python -m rank_validation --records <JSON> --output <新規local-output>` で比較できます。

JSONの最上位は `records`, `official`, `unresolved`, `experiments`, `provenance`。各recordには `model_id`, `experiment_id`, `case_key`, `release_id`, `evaluation_sha256`, `scorer_id`, `score_1`～`score_5` が必要です。出力の `normalized_records.json` が具体例です。`evaluation_sha256` は正解・設定・採点器の対応を示すもので、公開入力の同一性を代替しません。別途入力・予測の原本ハッシュを記録してください。

現在のZIPには攻撃に渡した公開入力の一部がないため、保存時の入力対応は引き継ぎ索引に依存します。原本ハッシュと再採点の成功を、未知人物評価やすべての公開入力の一致の証明として扱いません。

信頼できる順位推定へ進むには、公開特徴で不足した加工条件を持つ正解付き確認問題を別に作り、生成方式・比較ゲートを先に固定してから新候補を評価する必要があります。既存R版を増やしたり同じ集団を30回再抽出したりしても、加工方式の不足は埋まりません。予備戦実問題は別の確認パネルとして保持し、本戦の模擬得点と平均しません。

## 検証と独立した入口

```powershell
Set-Location main_round/mock_test
<既存Python> -B -m unittest discover -s rank_validation/tests -v
<既存Python> -B -m rank_validation.replay --handoff <展開先> --output local-output/<新しい検証名>.json
<既存Python> -B -m rank_validation.public_profiles --root <repo> --output local-output/<新しい診断名>
```

初期監査では46テスト、公開プロファイル8検証、通しの81ペア比較を実施しました。その後の共同評価・生成器・来歴検証には追加テストがあります。厳密再採点では期待する全22対象または単一対象、1,049候補、326問、所属ID・φを検証しました。実行環境は既存Python3.12.14、NumPy2.4.6、pandas2.3.3。公式キットの29依存版を検証します。

初回検証時点ではGitHub保存前でした。今回の公開対象は、この評価システムと選択した検証記録です。公式提出・試行消費は行っておらず、固定M/T/R、公式キット・配布入力、過去の評価記録は変更していません。

公式資料：[PWS Cup 2026](https://www.iwsec.org/pws/2026/cup26.html)、[固定版公式採点コード](https://github.com/pwscup/pwscup2026-kit/blob/main-attack-20260928/starter/src/pwscup2026/scoring/attack_score.py)。公式サイトの更新では攻撃締切は2026年10月11日21:00 JSTへ延長されています。

## 本番と同じ4モデルの共同評価

`adapters.py` と `model_worker.py` は、公開入力だけを受け取り、凍結された Rv1 / Rv2 / pl1 / Tv105 を本番と同じ処理で実行します。全22対象、既知の自己専有人物163人の処理、Rv1の対象をまたぐ中央値、pl1の共同割当、Tv105の固定seedによる元のルーター生成手順を維持します。Tv105のルーターは渡された公開背景から作るため、重み自体は問題ごとに変わります。モデル本体・依存ハッシュの判定条件は変更しません。

実際の本戦公開入力から再生成した4モデル8CSVは、保存提出用CSVと全数値が完全一致しました。Rv1/Rv2/pl1はバイトも一致し、Tv105だけCRLF/LFの改行差があります。証拠は [公開用検証記録](evidence/publication/publication_manifest.json) のparity欄に保存しています。これは保存提出用CSVとの一致であり、運営が保管する実アップロードのハッシュ照合ではありません。

`batch_worker.py` はRv1/Rv2/Tv105の共通T88成分を再利用します。各対象で正規のRv2後処理と一致を検査し、対象をまたぐRv1の計算とTv105のルーターを保持します。各モデルの出力には公開入力・ソース・依存資産・予測のハッシュを記録します。pl1は独立したworkerで実行します。

### 人口構成を含めた固定確認問題

`generation_v3.py` は共有関係、希少者の出生ラベル、打切り時刻を保った22対象の共同問題を生成し、実際の公開C・質問・背景の6指標だけで26通りの加工候補から選びます。攻撃得点や公式結果を加工選択に渡しません。

公開人口構成の監査で、実本戦の23質問ファイルはそれぞれ326人全員が同じ県に属する一方、旧生成器の私的人物は県が分散することが判明しました。公式ソースにも私的コアによる県分布の偏りが明記されています。v3の攻撃得点を見る前に、これを修正する `generation_v4.py` を別プロトコルとして固定しました。

v4は、会員・対照の両方を同じ県・性・5歳階級構成へ対称に移します。元ペアと移動先ペアの平均年齢に基づく割当を用い、共有人物の値、所属ID、出生希少ラベル、発症・死亡・時刻は変更しません。4対象の奇数度数セルは最小の均衡丸めで対称化し、差分を保存します。これは人口構成の輸送近似であり、未知の県別の臨床分布を復元したものではありません。

固定プロトコルは [generation_v4_protocol.json](generation_v4_protocol.json)。2026-10-09 04:35:57 UTCにモデル得点の確認前に固定し、SHA256は `c383c5fe06136fdbdd11d501467c7c4731919137cf2c8b7b8d701a524dc95038`。開発seedは2026100941、確認seedは2026100951/2026100952です。都合のよいseedや対象だけを選ばず、公開分布の不適合も保持します。異なる乱数でも供給元の母集団は共通であり、未知人物による外部検証とは区別します。

`panel.py` は各問題の公開入力と非公開正解を別フォルダに保存し、公式方式のAIA補正値を1,000回の標本抽出で一度だけ計算します。同じ問題の全モデルがこの正解・補正値を共用し、予測workerには公開フォルダだけを渡します。Pythonの読取監査は誤参照検出であり、悪意のあるコードを隔離するOSサンドボックスではありません。

### 再実行と新候補の評価

保存先は常に `mock_test/local-output` の新しいフォルダです。以下のパスはリポジトリ直下から指定します。

```powershell
# 固定されたケースを生成（既にある場合は保存済みのものを利用）
.\main_round\mock_test\rank_validation\run_joint.ps1 -Action Generate -Generation v4 -Seed 2026100941 -Output main_round/mock_test/local-output/new-run/case
# 同じ問題用の共通補正値を計算
.\main_round\mock_test\rank_validation\run_joint.ps1 -Action Prepare -Case main_round/mock_test/local-output/new-run/case -Output main_round/mock_test/local-output/new-run/panel
# 検証済みの共通計算でRv1/Rv2/Tv105を実行して採点
.\main_round\mock_test\rank_validation\run_joint.ps1 -Action Batch -Panel main_round/mock_test/local-output/new-run/panel
# pl1を同じ問題で採点
.\main_round\mock_test\rank_validation\run_joint.ps1 -Action Predict -Panel main_round/mock_test/local-output/new-run/panel -Model pl1
```

新しいモデルは `predict_world(candidates, self_labels, background, self_challenge, targets, schema, cfg)` を持つプラグインを用意し、`F_mia` / `F_aia` のDataFrameを返します。`targets[k]` には `release` と `challenge` だけが入ります。正解・seed・加工名を受け取りません。固定済みの追加学習資産がある場合は、プラグインに `FITTED_ASSETS` を宣言してハッシュを検証します。詳細は `plugins.py` を参照してください。

```powershell
# 同じ固定v5確認2ケースで新候補を評価しRv2との差を表示
.\main_round\mock_test\rank_validation\run_candidate.ps1 -Plugin <model_plugin.pyのパス> -Name candidate_v1
```

実行済みのモデル名・予測・得点は上書きしません。新候補の実行前に4基準モデルの全ケース得点が揃っていることを検査します。閲覧用 `report.html`、成分別 `comparison.csv`、入力検証と既知4モデルの順位照合を含む `result.json` を作ります。

`joint_report.py` は公式画像の丸め区間を考慮し、4モデルの6通りの大小関係をケースごと・確認平均で照合します。既知モデルの順位一致と、未知モデルの順位予測の検証は別の判定です。既知順位が合わない間は模擬順位を本番の予測順位として採用しません。予備戦の20対象評価は `prelim_panel.py` で別に実行し、本戦合成得点と平均しません。

## 固定した比較指標と新しい確認ケース

v4では公式方式の合計順位がケースごとに変動しました。確認2ケースの平均も6組中3組だけ一致。MIAとAIAの1係数補正も、モデルを1つ外す4検証すべてで完全一致に失敗したため採用していません。結果は `local-output/ranking-v4-20261009/raw-rank-report/` と `ordinal-diagnostic/` に保存しています。

`ranking_metric.py` の比較指標は **全体MIA + 補正済みAIA** の対象別等重み平均です。希少者MIAによる上乗せは `rare_lift` として別に表示します。係数の学習、対象の重み付け、モデル名ごとの補正は行いません。公式方式の総合点もそのまま保存します。希少者だけを改善するモデルは、この指標の単独評価に適しません。

v4の3ケースは指標の設計資料とし、検証成功の根拠には使いません。v5は生成処理をv4と同一に保ち、未使用seed **2026100961 / 2026100962** を生成前に予約しました。主判定は、この2ケースの等重み平均で既知4モデルの6比較がすべて一致すること。同点は失敗です。個別ケースの逆転、各差の範囲、対象を1つ外した順位は必ず別に表示します。

新候補には同じ保存済みv5ケースを使います。`run_candidate.ps1` は全基準モデルの完了を確認し、候補の予測・採点・比較レポートを作ります。正の差が候補の改善を示しますが、ケースや対象の除外で符号が変わるときは結果を混在と表示します。これは比較の判断材料であり、本番提出の指示ではありません。

v5の新規生成は `run_joint.ps1 -Action Generate -Generation v5 -Seed 2026100961`（もう1ケースは末尾62）で行います。`-Output` に新しいケース保存先を付け、Prepare / Batch / Predict を上の手順で実行します。4モデルが揃った2パネルは `-Action Report -Generation v5 -Panels <panel61>,<panel62> -Output <新規report先>` で照合できます。固定した指標とfreeze記録の原本は `evidence/v5/` に保存しています。

評価条件・公式採点コード・実行環境・モデルの記録された依存情報を署名で照合し、異なる版は結合しません。元の保存ケースとパネルの公開入力・正解・seedも結び付けて検証します。旧結果への署名追加は予測CSVを再採点して行い、元の12結果と対象別CSVが完全一致することを確認しています。

残る制約として、供給元の母集団は共通で、新ケースは未知の実人物による外部検証ではありません。v5の両ケースとも公開特徴の範囲外が11/22対象あります。実際の他チーム所属正解・AIA正解・運営の補正値は非公開です。公開分布への近似や既知順位の一致から、これらが復元できたとは判断しません。
