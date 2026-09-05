# OpenDP AIM 匿名化ワークエリア

PWS Cup 2026向けに、OpenDPのAIM（Adaptive Iterative Mechanism）を使った
差分プライバシー合成データを試作する独立プロジェクトです。

公式採点環境である `starter/` の依存関係とロックファイルは変更しません。
OpenDP用の仮想環境はこのディレクトリの `.venv/` に分離し、生成データは
リポジトリ直下の `local-output/opendp_aim/` に出す方針です。

## 構成

```text
opendp_aim/
├── config/
│   ├── pwscup_aim_v1.json # 初期ベースライン
│   ├── pwscup_aim_v2.json # 公開希少ドナー後処理
│   ├── pwscup_aim_v3.json # 公開転帰ドナー後処理
│   └── pwscup_aim_v4.json # 公開希少率で希少群を安定化した既定設定
├── scripts/
│   ├── generate_pwscup.py # B_practice.csvからC.csvを生成
│   ├── toy_attack.py      # 既存Cへの非公式toy MIA/AIA自己診断
│   └── smoke_test.py      # 玩具データでの動作確認
├── pyproject.toml    # OpenDP専用の依存定義
├── uv.lock           # 解決済み依存関係
├── setup.ps1         # 専用仮想環境を同期
├── run-smoke.ps1     # 実データを使わないAIM動作確認
├── run-toy-attack.ps1 # 既存Cへのtoy攻撃（Cは再生成しない）
└── run-pwscup.ps1    # 生成・採点・ZIP検証を一括実行
```

## セットアップ

リポジトリ直下から実行します。

```powershell
powershell -ExecutionPolicy Bypass -File .\opendp_aim\setup.ps1
```

既存のリポジトリ内蔵 `uv` とPython 3.12を使います。PC全体のPython環境や、
`starter/.venv` には変更を加えません。

`pandas` は、Private-PGMが合成結果を実体化するときに必要ですが、現行の
`opendp[mbi]` では推移的依存として導入されないため、このプロジェクトで明示的に固定しています。

## AIMの動作確認

```powershell
powershell -ExecutionPolicy Bypass -File .\opendp_aim\run-smoke.ps1
```

このテストは小さな玩具データにAIMを適用し、同じ列・行数の合成データを生成します。
`AIM smoke test: OK` と表示されれば、OpenDP・MBI・合成処理が動作しています。

## 一括実行

リポジトリ直下から次の1コマンドを実行します。

```powershell
powershell -ExecutionPolicy Bypass -File .\opendp_aim\run-pwscup.ps1
```

このコマンドは、AIMによるDPリリース、`C.csv`生成、提出ZIP検証、公式自己採点、
JSON保存までを順番に実行します。練習用トークンを使うため、本番提出時は
`-Token`へ自分のCodaBenchトークンを指定してください。

既定では`pwscup_aim_v4.json`を使います。v3はv2の希少群後処理に加えて、公開
`A_bg.csv`だけから固定した発症・死亡Cox係数でDP合成共変量を2次元の公開リスク空間へ
写像し、その最近傍行から`time`、`onset`、`death`をまとめて移植します。これにより、
個別のBレコードの転帰を残さず、公開背景データにある集団レベルの共変量・転帰関係を
再現します。最近傍探索はDP出力と公開データだけを使う決定的な後処理であり、追加の
プライバシー予算やCoxライブラリを使いません。公開係数と死亡リスク重みは設定に固定しています。

v4はAIM本体と公開Cox転帰後処理をv3のまま維持し、希少群の件数と結合分布を安定化
します。公開`A_bg.csv`の希少率を公開出力行数へ掛けて目標件数を決め、DP合成出力を
公開Mahalanobis距離で順位付けして上位行を対象にします。対象行の希少マーカー7列は、
公開`A_bg.csv`の`4 <= 距離 < 5.5`にある行から固定順でまとめて移植します。目標外の
希少行は公開閾値直下の非希少行で修復します。現在の公開データと1049出力行では目標は
63件です。件数、対象選択、ドナー選択のすべてが公開データまたはDP出力だけに依存する
後処理なので、追加のプライバシー予算は消費しません。

```powershell
powershell -ExecutionPolicy Bypass -File .\opendp_aim\run-pwscup.ps1 `
    -Token "YOURCODEBENCHTOKEN"
```

すでに生成・採点した`C.csv`を維持し、本番トークンだけ差し替えてZIPを作り直す場合は、
新しいDPリリースを発生させないよう`-SkipGenerate`を併用します。既存C、設定、privacy
reportのSHA-256が一致しない場合は安全のため停止します。

```powershell
powershell -ExecutionPolicy Bypass -File .\opendp_aim\run-pwscup.ps1 `
    -SkipGenerate `
    -Token "YOURCODEBENCHTOKEN"
```

## 一つずつ実行

### 1. AIMでC.csvを生成

```powershell
$env:UV_PYTHON_INSTALL_DIR = (Resolve-Path '.\.tools\python').Path
$env:UV_CACHE_DIR = (Resolve-Path '.\.uv-cache').Path
$env:VIRTUAL_ENV = $null

& .\.tools\uv\uv.exe run `
    --project .\opendp_aim `
    --frozen `
    python .\opendp_aim\scripts\generate_pwscup.py `
    --input .\participant_data\B_practice.csv `
    --background .\participant_data\A_bg.csv `
    --schema .\participant_data\schema.json `
    --config .\opendp_aim\config\pwscup_aim_v4.json `
    --output .\local-output\opendp_aim\C.csv `
    --privacy-report .\local-output\opendp_aim\privacy_report.json
```

外側の仮想環境を有効化したまま直接`uv`を実行すると、`VIRTUAL_ENV ... will be ignored`
という警告が出ることがあります。処理上は無害です。上の`$env:VIRTUAL_ENV = $null`で
子プロセスに対する警告を抑制できます。

### 2. 有用性を自己採点

```powershell
& .\.tools\uv\uv.exe run `
    --project .\starter `
    --frozen `
    python -m pwscup2026.kit.selfscore `
    .\local-output\opendp_aim\C.csv `
    --dist .\participant_data
```

### 3. 提出ZIPを作成

```powershell
Copy-Item `
    .\participant_data\sample_submissions\token.txt `
    .\local-output\opendp_aim\token.txt `
    -Force

Compress-Archive `
    -LiteralPath @(
        '.\local-output\opendp_aim\C.csv',
        '.\local-output\opendp_aim\token.txt'
    ) `
    -DestinationPath .\local-output\opendp_aim\defense_submission.zip `
    -CompressionLevel Optimal `
    -Force
```

### 4. 提出ZIPを公式検証

```powershell
& .\.tools\uv\uv.exe run `
    --project .\starter `
    --frozen `
    python -m pwscup2026.kit.validate `
    .\local-output\opendp_aim\defense_submission.zip `
    --kind defense `
    --dist .\participant_data
```

## PWSCUPデータフロー

```text
participant_data/B_practice.csv
  -> 公開スキーマとA_bgに基づく固定ビン化
  -> Fixedで転帰周辺をDP測定
  -> OpenDP AIMで残りのDP周辺分布を適応測定
  -> Private-PGMによる分布推定（後処理）
  -> A_bgによる数値復号・公開制約による整形（後処理）
  -> A_bg希少率に合わせ、DP出力の希少候補上位行を公開ドナーで行単位復号（後処理）
  -> 公開Coxリスク空間の最近傍A_bg行からtime/onset/deathを一括移植（後処理）
  -> local-output/opendp_aim/C.csv
  -> starter環境の公式自己採点器
```

`generate_pwscup.py`が行う`.release()`は1回です。採点、ZIP作成、DP分布からの合成、
公開データを使った復号は後処理です。本番データで生成を複数回行う場合は、各リリースの
プライバシー予算を合成して管理する必要があります。

## 既存v4へのtoy攻撃

現行の`local-output/opendp_aim/C.csv`を再生成せず、自己診断用のMIAとAIAを実行します。
MIAは採点説明notebookの1-NN攻撃と配布参照実装のDOMIAS KDEを使います。DOMIASは
公式サンプルと同じ全`A_bg`参照、およびtoy非会員を参照から外したcross-fit感度分析の
両方を測ります。AIAは配布参照実装のQI最近傍攻撃です。攻撃感度の確認用に、生の
`B_practice.csv`をCとみなしたpositive controlと、Cを一切見ず全員へ`time=10`と答える
public baselineも同時に測ります。

```powershell
powershell -ExecutionPolicy Bypass -File .\opendp_aim\run-toy-attack.ps1
```

結果は`local-output/opendp_aim/toy-attack-v4/`のJSONとCSVへ保存されます。この処理は
既存Cの読み取りだけを行うため、追加のDPリリースやプライバシー予算消費はありません。
ただし、手元にない事務局専用のmembership/AIA真値の代わりに、B全体と公開A_bgから
候補・対照を構成した非公式診断です。特にAIAの対照はv4の復号・転帰ドナーにも使った
`A_bg`なので、対照成功率が高くなり漏洩指標が小さめに出る可能性があります。JSONには
希少定義の感度分析、MIAのローカル偶然水準、入力・設定・実行環境のハッシュも保存します。
大会の正式な保護スコアや攻撃得点ではありません。
