# mock_TS_using_prelim

予備戦の**実際の配布データ・運営公開正解**を使い、自分たちの攻撃をMIA・AIAの公式指標で評価します。名前にmockが入っていますが、この環境では問題や正解を人工生成しません。自チームは予備戦の **team24**、攻撃対象は自己を除く **20チーム**です。

必要なデータと予備戦の公式コードをこのフォルダーにコピーしました。実行時にDownloadsや元の予備戦フォルダーを参照しません。Python環境は既存の `main_round/attack/.venv` を使います。既存の `mock_TS`・`retest`・各作業者の攻撃は変更しません。

## 最初に動かす

リポジトリ直下から実行します。

```powershell
# 21チームの運営公開回答を再採点し、公表値と照合
.\main_round\mock_test\mock_TS_using_prelim\run.ps1

# データと公式コードのハッシュ照合だけ
.\main_round\mock_test\mock_TS_using_prelim\run.ps1 -Action Verify

# 付属の公式参照攻撃を20標的へ適用して採点
.\main_round\mock_test\mock_TS_using_prelim\run.ps1 -Action Run
```

結果は `main_round/mock_test/local-output/mock_TS_using_prelim/<実行日時>-<処理>/` に保存します。`report.html` はブラウザーで開ける日本語レポートで、表の見出しを押すと並べ替えられます。CSVとJSONにも数値・明細・入力ハッシュを保存します。既存の非空出力先を上書きしません。

このPCでの確認済みレポート：

- [全21チームの公表結果との照合](verification_reports/published_replay/report.html)（[比較CSV](verification_reports/published_replay/comparison.csv)）
- [公式参照攻撃の実行結果](verification_reports/reference_baseline/report.html)（[標的別CSV](verification_reports/reference_baseline/by_target.csv)）

HTMLはダウンロードしてブラウザーで開けます。GitHub上ではCSV・JSONも参照できます。

## 何をどこに置いたか

```text
mock_TS_using_prelim/
├─ run.ps1 / cli.py          実行入口
├─ example_attack.py        動く公式参照攻撃・自作攻撃のひな型
├─ scoring.py               厳密なID確認と公式採点の呼出し
├─ runner.py / worker.py     攻撃用入力の書出し・別プロセス実行
├─ report.py                HTMLレポート
├─ data/
│  ├─ public/
│  │  ├─ team24_processing/ 匿名化フェーズに実配布された原本6件
│  │  └─ attack_targets/    攻撃フェーズの全21C・全21問題と一覧等45件
│  ├─ private/ground_truth/ 運営の正解：22pool表・21AIA正解・説明
│  └─ reference/
│     ├─ attack_submissions/ 21チームの原本ZIPと展開済み回答
│     ├─ published_results/ 結果Excel・全シートCSV・チーム対応表・明細
│     └─ own_submissions/   自分たちの匿名化v5・攻撃v9の最終提出
├─ official-kit/             固定版公式コード・依存一覧・ルールブック・LICENSE
├─ provenance/               出典、元パス、SHA-256、コピーと照合の記録
└─ tests/                    採点・ID・入力分離の検証
```

データは178ファイル、公式資料・ソースは36ファイルです。詳しい読み方は [data/README.md](data/README.md)、出典は [provenance/README.md](provenance/README.md) と [data_manifest.json](provenance/data_manifest.json) を参照してください。CSV派生物と原本コピーを区別し、原本のバイト列を保持しています。

`public` は「攻撃時にteam24が使用できた情報」を意味します。自チームの `B_self.csv` は正規の配布情報なので攻撃入力に含みます。相手の `pool_ids`・`aia_truth`、各チームの回答、結果表は採点・比較側だけで使います。自己のC24とAIA問題も保管し、自己攻撃は集計から除外します。

2026-10-07の追加依頼により、このフォルダーのデータ178件と公式資料36件を既存の非公開GitHubへ保存する対象にしました。`data/.gitignore` は台帳にあるファイルだけを許可します。認証トークンは評価に必要ないためコピーしていません。初回の出典台帳にある `local_only` と初回検証の未公開記載は、その時点の履歴です。今回の保存範囲と検証は [publication_20261007.json](publication_20261007.json) を参照してください。

## 自作の攻撃を実行する

攻撃の実装ファイルは担当者の作業フォルダーに置き、次の関数を用意します。[example_attack.py](example_attack.py) が実行可能な例です。

```python
def predict_world(*, candidates, self_labels, background, self_challenge,
                  targets, schema, cfg, attacker_id=24,
                  self_release=None, utility_reference=None):
    # targets[k] = {"release": C_k, "challenge": aia_challenge_k}
    # candidatesのrecord_idと、各challenge_row_idを保持する。
    # この関数に渡された予備戦入力だけを使用する。
    return {"F_mia": mia_dataframe, "F_aia": aia_dataframe}
```

`self_release`・`utility_reference`・`attacker_id` は必要な場合に受け取れます。既存の7引数形式も呼び出せますが、**標的や自己番号を本戦用に固定したコードは予備戦対応のアダプターが必要です**。たとえば `Attack_R/sugiyama_adapter.py` は本戦の22標的を前提としているため、そのまま指定できません。凍結済みの攻撃本体を変更せず、予備戦の20標的・team24を扱う別アダプターを用意してください。

```powershell
.\main_round\mock_test\mock_TS_using_prelim\run.ps1 -Action Run `
  -Plugin .\main_round\attack\自分のフォルダー\prelim_adapter.py
```

プラグインは、新しいPythonプロセスで、書き出した攻撃用入力を作業場所として実行します。正解や他チームの回答は引数に渡しません。ただし、信頼するローカルPythonコードの実行であり、OSでファイルの読み取りを禁止する仕組みではありません。プラグインから正解ファイルを読んだり、予備戦正解で学習済みの状態を持ち込んだ場合は、未知データに対する評価にはなりません。

## 別プログラムで作った回答を採点する

入力の書き出し、攻撃、採点を別々に実行できます。以下は同じ入力を使い回す例です。実在する自分のプラグインへパスを置き換えてください。

```powershell
$prelimBase = '.\main_round\mock_test\local-output\mock_TS_using_prelim'
$prelimTool = '.\main_round\mock_test\mock_TS_using_prelim\run.ps1'

& $prelimTool -Action Export -Output "$prelimBase\inputs-01"

& $prelimTool -Action Attack `
  -BundleDirectory "$prelimBase\inputs-01\public" `
  -Plugin .\main_round\mock_test\mock_TS_using_prelim\example_attack.py `
  -Output "$prelimBase\attack-01"

& $prelimTool -Action Score `
  -PredictionDirectory "$prelimBase\attack-01\predictions" `
  -Output "$prelimBase\score-01"
```

`Attack` の代わりに別のプログラムで2本のCSVを作り、その保存先を `-PredictionDirectory` に指定しても採点できます。

| ファイル | 必要な内容 |
|---|---|
| `F_mia.csv` | `record_id` と20標的の番号の列。候補は実team24の1,049人、確信度は0〜1 |
| `F_aia.csv` | `target_k,challenge_row_id,time_hat`。20標的×326行、予測年数は0以上の有限値 |

重複・欠落・余分なID・非整数ID・NaN・infを拒否します。CSVの行順、MIA標的列の順序は変えられます。自己24の列・行が追加されていれば自己分を除外します。他ラウンドの回答を流用してはいけません。外部CSVだけから生成に使った入力を証明することはできないため、その場合は予備戦IDへの対応を利用者の申告として記録します。付属Runnerの回答では入力・出力ハッシュも照合します。

## 得点の意味

採点には `prelim-attack-20260825` の公式関数をコピーして使います。公式GitHubの固定版と全36ファイルのGit blobを照合しました。Windows改行の差は確認時のみ正規化し、保存バイト列は元ファイルのままです。

| 表示 | 計算 |
|---|---|
| MIA力 | 標的ごとにFPR=1%でのTPRを全体・希少層で計算し、大きい方を採用。その20標的平均 |
| AIA力 | `time_hat` が正解の±0.5年以内かを判定。標的・対照の成功率差を正規化し、全体・希少層の大きい方を20標的平均 |
| 攻撃力・score_1 | MIA力＋AIA力（0〜2） |
| score_2 / score_3 | MIA全体／希少の内訳 |
| score_4 / score_5 | AIA全体／希少の内訳 |

AIA希少層には公式の1.64×SE補正を適用します。予備戦には本戦の標的別 `φ_i` 控除はありません。希少MIAに所属者がいない標的の値は未定義として保持し、公式と同じ集約を使います。内訳の平均値に後からmaxを取っても、公式の攻撃力は復元できません。

これは攻撃者側のMIA・AIA評価です。防御者の有用性U・匿名性Anon・最終総合得点の再計算機能ではありません。結果表のそれらの値は原本として保存しています。

## 公表結果との照合

`Replay` は21チームの公開回答を攻撃者ごとのID表で再採点し、以下を確認します。

- 21件の攻撃力合計
- MIAの420組：全体・希少TPR、採点値、AUC、所属者数
- AIAの419組：採点値、全体・希少の値、標的・対照成功率

2026-10-07の初回照合で数値5,589件が許容差10⁻¹⁰以内で一致し、最大誤差は約5.55×10⁻¹⁷でした。team24はMIA **0.15792307692307692**、AIA **0.18274506769014637**、合計 **0.3406681446132233** を再現しました。

運営原本ではcohort04のAIA標的23が欠落し、公表値は19標的平均です。`Replay` だけでこの履歴を再現し、欠落を0で補いません。任意のCSVでこの例外を使うことはできません。通常の新作評価では20標的すべてが必要です。

## 検証と移動

```powershell
.\main_round\attack\.venv\Scripts\python.exe -I -B -m unittest discover `
  -s main_round/mock_test/mock_TS_using_prelim/tests -v
```

別PCではこの非公開リポジトリを取得すれば、実データを含むこのフォルダー一式がそろいます。Python環境は `main_round/attack/setup.ps1` で準備するか、Python 3.11以上の環境へ [requirements-kit.txt](official-kit/starter/requirements-kit.txt) の依存を導入してください。`run.ps1 -Python <python.exe>` で環境を指定できます。Downloadsや元の予備戦保存先は不要です。

この問題を繰り返し使って調整した結果は、本戦での未知データ性能とは区別します。本戦との違いや、手元の攻撃が既に予備戦データを学習・調整に使用したかを、比較記録に残してください。
