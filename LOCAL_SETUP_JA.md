# PWSCUP2026 ローカル実行ガイド（Windows）

この追加ファイル群は、PC 全体へ Python や Docker を入れず、リポジトリ内だけで
スタートキットを実行するためのものです。`uv`、Python 3.12、仮想環境、キャッシュは
すべてこのフォルダー内へ保存されます。

## 1. 環境を作る

PowerShell を開き、次を実行します。初回だけインターネット接続が必要です。

```powershell
Set-Location "C:\path\to\pwscup2026-kit"
powershell -ExecutionPolicy Bypass -File .\setup-local.ps1
```

この PC ではすでに基本環境の構築が完了しているため、通常は次の手順から再開できます。

## 2. 防御・自己採点・攻撃・ZIP検証を一括実行する

```powershell
powershell -ExecutionPolicy Bypass -File .\run-practice.ps1
```

処理内容は次のとおりです。

1. `my_solution/run_defense.py` で `C.csv` を生成
2. 有用性 `U` と4観点を自己採点
3. 防御提出 ZIP を作成し、公式検証器で検証
4. 公式参照 MIA / AIA で攻撃 CSV を生成
5. 攻撃提出 ZIP を作成し、公式検証器で検証

成果物は `local-output/` に出ます。

```text
local-output/
├── C.csv
├── F_mia.csv
├── F_aia.csv
├── token.txt
├── defense_submission.zip
└── attack_submission.zip
```

引数なしでは配布サンプルの練習用トークンを使います。本番提出用 ZIP を作るときだけ、
CodaBench で割り当てられた自分のトークンを指定してください。

```powershell
powershell -ExecutionPolicy Bypass -File .\run-practice.ps1 -Token "自分の英数字トークン"
```

## 3. 防御だけを素早く試す

攻撃生成を省くと、匿名化方式の調整と自己採点を短いループで回せます。

```powershell
# 既定の共変量条件付き生存時間モデル
powershell -ExecutionPolicy Bypass -File .\run-practice.ps1 -SkipAttack -DefenseMethod copula_survival -Seed 0

# 比較用の素朴なコピュラ合成
powershell -ExecutionPolicy Bypass -File .\run-practice.ps1 -SkipAttack -DefenseMethod copula_synth -Seed 0
```

自分の方式を実装する入口は `my_solution/run_defense.py` の `build_defense` 関数です。
入力 `b_df` と同じ行数の DataFrame を返してください。最終的な値域調整、列順、
`record_id` の除去は同ファイル内の後処理が行います。

## 4. ノートブックを開く（任意）

JupyterLab は追加依存なので、必要なときだけ入れます。

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-local.ps1 -Notebook
& .\.tools\uv\uv.exe run --project .\starter --frozen --extra notebook jupyter lab .
```

ブラウザーで `notebooks/はじめてのPWSCup.ipynb` または
`notebooks/採点のしくみ.ipynb` を開いてください。

## 5. 覚えておく制約

- ルールと採点定義は `rulebook/` の PDF が正本です。
- 手元で完全に再現できる得点は有用性だけです。保護と攻撃力は他チーム情報が必要です。
- 防御 ZIP の直下は `C.csv` と `token.txt` だけです。
- 攻撃 ZIP の直下は `F_mia.csv`、`F_aia.csv`、`token.txt` だけです。
- `C.csv` は配布 B と同じ行数で、`record_id` を含まない13列です。
- 本番提出前は必ず、実トークンを指定して公式検証器が `OK` になることを確認してください。

環境を消す場合は `.tools/`、`.uv-cache/`、`starter/.venv/` を削除すると元に戻せます。
