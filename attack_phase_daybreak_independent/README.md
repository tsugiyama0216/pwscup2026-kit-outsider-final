# Daybreak Blue 独立攻撃モデル

PWSCUP2026予備戦・team 24専用の、v1/v2/v3とは別実装のMIA/AIA生成フローです。
モデル識別子は `gpt-daybreak-blue-latest`、アルゴリズム識別子は
`daybreak_blue_reciprocal_conformal_integrity_router_v1` です。

## 既存方式との違い

- productionコードは `attack_phase`、`attack_phase_v2`、`attack_phase_v3`、
  `attack_phase_daybreak` をimportしません。
- MIAはDOMIAS/KDE、Sinkhorn、人数割当を使わず、4つの頑健viewにおける相互最近傍、
  view合意度、既知負例conformal校正、target間marginで順位を作ります。
- AIAは一対一Hungarian連結を使わず、6部分空間の最近傍合議で再利用可能なcodebookを作ります。
  `A_bg`で学習したQI→eventモデルによる公開CのAUCが0.60未満（または測定不能）のtargetだけ、
  残差補正expertからcensoring-horizon expertへtarget単位で切り替えます。
- v1/v2/v3の候補CSVや診断値を入力にしません。

## 実行

リポジトリ直下で次を実行します。

```powershell
powershell -ExecutionPolicy Bypass -File .\attack_phase_daybreak_independent\run_daybreak_independent.ps1
```

公式名の2 CSVは
`submissions/prelim/team24/attack-candidate-daybreak-independent/` に生成されます。
識別情報とhashは
`local-output/prelim/team24/attack-daybreak-independent/model_manifest.json` に保存します。

## 安全境界

入力は配布済みローカルCSVだけです。ネットワーク、認証、token読取り、ZIP作成、
サーバー操作、提出処理は実装していません。候補ディレクトリには `F_mia.csv` と
`F_aia.csv` 以外を置きません。

## 評価上の注意

事務局限定の真値は配布されていないため、公式得点はローカルでは算出できません。
`evaluate_shadow_daybreak_independent.py` の結果は方式比較用の疑似真値テストであり、
リーダーボード得点の推定ではありません。未使用2 seedの3 stress条件では、MIA平均が
`0.9811`（v3参照 `0.9636`）、AIA平均が `0.8613`（v3参照 `0.8288`）でした。
ただしAIAはnoise 0.10単独ではv3参照を下回り、time-shuffledで大きく改善するtrade-offがあります。
独立方式は比較候補として保存し、v1/v2/v3を上書きしません。
