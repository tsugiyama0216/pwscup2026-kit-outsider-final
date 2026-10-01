# 本戦の共通模擬問題生成・評価器

**攻撃アルゴリズムを、正解が分かる模擬問題に適用して比較する道具です。** 公式と同じ採点関数を使います。本戦22標的の予測CSVから公式点を推定するものではありません。

2026-09-30に攻撃実装から分離し、`main_round/mock_test/` に置きました。M・TSのどちらの攻撃も、共通の公開入力APIで同じ問題に適用できます。Pythonパッケージ名は互換性のため `local_eval` を維持します。短い開始手順は [共通ツールの入口](../README.md) を参照してください。

## 別PCで模擬問題を生成する

この非公開リポジトリには元になるteam10の`A_bg.csv`・`B.csv`・`schema.json`と固定版公式キットがあります。別PCでcloneした後、Windows PowerShellで`main_round/mock_test/`へ移動します。

```powershell
..\attack\setup.ps1
.\generate.ps1 -Split dev -Seed 20260927 -Count 3
# 6抽出に増やす例： .\generate.ps1 -Split dev -Seed 20260927 -Count 6
# 単一問題・1方式だけ： .\generate.ps1 -Split holdout -Seed 971001 -Count 1 -Style bootstrap
```

既定の `-Count 3` では開始seedから3回抽出し、各抽出から7加工方式を生成します。上の例は `local-output/shadow-cases/dev-20260927-n3/` に `suite.json` と抽出ごとのフォルダーを保存します。`-Count 1` の単一問題は `dev-20260927/` のように保存します。各問題の `public/` には加工C、背景、MIA候補、AIA質問、schemaだけを置き、`private/` には採点用の所属・時刻正解を分けます。元の人物行IDは出力しません。`manifest.json` に入力・生成コード・出力のSHA-256と抽出条件を記録します。既存の非空出力先は上書きしません。出力先はGit管理から除外されるため、模擬問題の実データを新たにGitHubへ載せません。

連続3seedで希少共有0・1・2人を各1回、6seedなら各2回含めます。手法を比較するときは同じ保存スイートを使用し、平均だけでなく問題別・方式別の成績と公式参照との勝敗を確認します。同じ問題の再採点では抽出の幅は増えません。

同じseed・分割・方式と同じ入力ファイルなら同じ模擬問題を再生成できます。`private/`を攻撃の`predict()`へ渡してはいけません。[第88候補の再現手順](../../attack/attack_TS/attack_research/published_v88/README.md)には固定25条件の採点確認もあります。

初版は2026-09-27に作成しました。現在のコードは本フォルダー、新しい共通ツールの出力は `main_round/mock_test/local-output/` に置きます。固定版公式キット・配布原本・提出済みデータは変更しません。

初回結果の保存先は `main_round/attack/attack_TS/local-output/local-eval-v1/dev-final-20260927/` でした。過去のTS/Mのモデル・予測・評価記録はそのまま保持し、この共通フォルダーへ移しません。これらのローカル実験出力は取得したPCにない場合があります。

## 実行

本戦の `main_round/mock_test/` で実行します。保存した問題を指定すると、同じ公開入力と正解を使って手法を比較できます。

```powershell
# 先にgenerate.ps1で保存した問題へMのv1を適用
.\evaluate.ps1 -CaseDirectory .\local-output\shadow-cases\dev-20260927-n3 -Plugin ..\attack\Attack_M\v1\model.py

# 複数の保存先も指定可能（各問題を先に-Count 1で生成しておく）
.\evaluate.ps1 -CaseDirectory .\local-output\shadow-cases\dev-20260927,.\local-output\shadow-cases\dev-20260928,.\local-output\shadow-cases\dev-20260929 -Plugin ..\attack\Attack_M\v1\model.py
```

Mのv1の既定は採用済み `full_reference_aia` です。攻撃関数へ渡すのは公開入力だけで、採点用の `private/` は評価器が読みます。本番だけの自己163人マスクはこのAPIには含みません。

`-CaseDirectory` を使うと、保存されたsplit・加工方式で公式参照とプラグインを比較します。`-Split`・`-Methods` は併用しません。`-Plugin` もカンマ区切りで複数指定できるため、同じ実行で複数手法を比較できます。

devで候補と設定を決めた後、固定した候補の確認には別の保存スイートを使います。

```powershell
.\generate.ps1 -Split holdout -Seed 20261001 -Count 3
.\evaluate.ps1 -CaseDirectory .\local-output\shadow-cases\holdout-20261001-n3 -Plugin ..\attack\Attack_M\v1\model.py
```

holdoutの結果を見て候補を変更した場合、その結果は以後の開発履歴として扱います。この人物分割は既に使用されているため、新しい抽出を増やしても完全に独立した人物検証にはなりません。

`-CaseDirectory` を省略すると、従来どおり指定splitの既定3抽出をその場で生成します。

```powershell
# 調整用データで、公式参照・多視点・定数・背景だけの4手法を比較
.\evaluate.ps1

# 自分の候補も同じ問題で比較する
.\evaluate.ps1 -Plugin .\local_eval\example_candidate.py

# 候補を固定した後にだけ、人物を分けた最終確認用で評価
.\evaluate.ps1 -Split holdout -Plugin .\local_eval\example_candidate.py
```

`-Output` で保存先を指定できます。保存先は `main_round/mock_test/local-output/` 内の空フォルダーに限り、以前の結果を上書きしません。`-NullRepetitions` は既定100。0でラベル入替診断を省略できます。処理は `main_round/attack/.venv/` の専用Python・CPU1スレッドで実行します。

既定7方式を含む保存問題1件なら主平均は1抽出×5方式の5条件、3抽出スイートなら15条件、6抽出なら30条件です。`-CaseDirectory` なしの自動生成評価は3抽出×5方式の15条件、v88の固定再現検査は5抽出×5方式の25条件です。抽出seedや方式が異なる結果をそのまま比較せず、同じ保存問題に各手法を適用してください。

初版作成時は最終確認用の構造・人数・分割だけを検査しました。その後、予備選手法の本戦移植でAIAを採点済みです。現在この人物分割を未使用の検証データとは呼べません。新しい研究でも、人物分割と抽出seedの使用履歴を区別して記録します。

## 出力の読み方

| 出力 | 用途 |
|---|---|
| `REPORT.md` | 点数・加工方式ごとの差・限界を読む |
| `summary.csv` | 評価した主対象方式の平均、最小最大、公式参照との同条件の差。既定自動生成では5方式×3抽出 |
| `results.csv` | seed・加工方式・手法ごとのMIA/AIA、全体/希少、偶然の水準 |
| `paired_comparison.csv` | 同一seed・方式での公式参照との差 |
| `by_style.csv` | 加工方式別の平均 |
| `by_case.csv` | 保存問題ごとの成績。特定の抽出に結果が偏っていないか確認 |
| `run.json` | 入力・実行コード・候補コードのハッシュ、固定条件、分割、所要時間 |

主指標は **MIA＋AIAのφ控除前代理値**（0〜2、高いほど攻撃が強い）です。

- MIA：FPR=1%のTPRを全体・希少層でそれぞれ計算し、大きい方。AUCは診断用です。
- AIA：time誤差±0.5年以内を成功とし、member/control成功率の差を正規化。希少側だけz=1.64の標準誤差補正を施し、全体と希少の大きい方。
- 運営の標的別補正値φは不明なので引きません。自チームφを他標的へ転用しません。防御側採点のMIA置換補正やK=12変換も、攻撃力の代わりに適用しません。
- ラベル入替100回の平均・95%点は参考です。公式の補正値、p値、信頼区間とは扱いません。

点数の高い候補を即採用するのではなく、加工方式別・同条件の差・希少群の少人数による振れ・対照手法を確認します。とくに希少member1〜2人のMIAは、1人の結果で大きく変わります。最終確認用の結果を見る前に候補と条件を固定します。

## 模擬問題の構成

元のA_bg 5,143人を、人物が重複しない調整用2,572人と最終確認用2,571人に固定分割します。県・性・5歳帯・発症・検出希少区分で対応するペアを各側へ確保します。分割seedは2026092701です。自B 1,049人を背景専用に使い、A_bgとの公開QIの重複がないことを検証します。

各抽出で以下を作ります。

- 模擬標的コホート：1,049人、検出希少24人。
- MIA候補：1,049人、標的と共有するmember26人、非member1,023人、検出希少24人。共有中の希少人数は0・1・2人の3条件。
- AIA：標的163人＋対照163人。各側の希少13・発症16・希少発症4人を固定。県・性・5歳帯・発症・希少の全項目を対応させ、対応できなければ条件を緩めず停止します。
- AIA標的はMIA共有26人と非重複。対照は模擬標的コホートとMIA候補のどちらにも入れません。
- 元のrecord_idを破棄し、予測用IDは無作為な行順の後に採番します。BとA_bgの同じrecord_idは、同一人物を意味しません。

調整用seedは20260927・20260928・20260929、最終確認用は20261001・20261002・20261003。同じ分割内での再抽出は独立データではありません。

| 加工方式 | 内容 | 主平均に含む |
|---|---|---|
| rounded | 年齢・検査値・timeを粗く丸める | はい |
| jitter | 背景SDの10%の数値ノイズ、time SD0.2年 | はい |
| strong_jitter | 背景SDの50%の数値ノイズ、time SD1年 | はい |
| bootstrap | 模擬標的から復元抽出 | はい |
| copula | 模擬標的の周辺分布・順位依存を学習し、新しい行を合成 | はい |
| copy | 無加工コピー。漏洩検出の確認 | いいえ |
| independent | 標的を使わず背景から生成。標的情報なしの確認 | いいえ |

これらは幅を持たせるための模擬方式で、他チームの加工法を特定・再現したものではありません。主平均の等重みは、本戦の各方式の割合を表しません。各方式の固定パラメータは [data.py](data.py) にあります。

## 新しい攻撃候補の追加

2026-09-27の履歴：予備選手法の本戦移植で、dev選択後にAIAのholdoutを採点済み。structured 0.636465、公式参照0.555723。以後この分割を未使用の検証用と扱わない。当時の `prelim_transfer` は現在の取得コードに未収録で、これらは保存された履歴値です。MIA v4の同時最適化は本APIの評価対象外。

[example_candidate.py](example_candidate.py) を別名でコピーし、次の関数を実装します。

```python
def predict(*, c, background, candidates, challenge, cfg):
    # c: 加工データ13属性
    # background: 独立した背景13属性
    # candidates: MIA候補 record_id + 13属性
    # challenge: AIA challenge_row_id + 公開QI7属性だけ
    # 必ず渡された行順を保つ
    return {"mia": confidence, "aia": time_hat}
```

`confidence` は長さ1,049の有限な0〜1、`time_hat` は長さ326の有限非負値。正解・希少ラベル・元の行番号・抽出seedは渡しません。AIAのBMI/FPG/time/onset等を別経路から参照してはいけません。返された長さ・値域・有限性を検査します。

このAPIは信頼できるローカル実装を比較するためのものです。Pythonモジュールは任意のファイルを読めるため、悪意あるコードを隔離する実行環境ではありません。外部ファイルや追加モデルを使う候補は、その入力も固定・記録してください。自動記録する候補ハッシュは渡したPythonファイル本体のみです。

この初版は1標的ずつの予測用です。全標的の同時最適化、自己challengeの163行を使う校正、既存予備戦の対象固定レシピには、別の入力設計が必要です。

## 主な限界

本戦の隠れた所属ラベル・AIA真値・φは取得していません。公開A_bgを模擬人物、自Bを背景に使うため、実戦と役割が逆です。希少区分も真のis_rareではなく公開検出器の代理です。B_selfの真ラベルを予測入力に混ぜず、未知の私的コホートや県偏りは再現しません。

公式関数と同じ数式でも、対象データが異なるため、得点の絶対値や本戦順位は予測できません。目的は**同じ模擬条件で候補の良し悪しや得意・不得意を比較すること**です。

## 検証

```powershell
..\attack\.venv\Scripts\python.exe -B -m unittest local_eval.tests.test_local_eval local_eval.tests.test_generate local_eval.tests.test_evaluate_case local_eval.tests.test_workspace_compat -v
```

分割非交差、AIAの厳密対応と人数、MIA希少構成、予測APIへの正解混入防止、入力変更の隔離、不正出力拒否、定数MIA=0.01、完全予測の点数方向、独立生成の標的不使用、コピー漏洩、コピュラ生成、入替診断の再現性を検証します。公式キットのファイルは変更せず、その関数を直接呼びます。
