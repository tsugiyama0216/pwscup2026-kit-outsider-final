# PWS Cup 2026 攻撃フェーズ作業領域

配布された競技CSVだけをローカルで読み、MIA/AIAの提出候補を生成します。ネットワーク、
CodaBench、他チームの非公開データにはアクセスしません。

## 実装した方式

- MIA: QI・全臨床値・転帰を使う複数のロバスト近傍スコアとDOMIASの対数密度比を統合。
  各候補が高くなりやすい標的横断の共通外れ値成分を除き、候補は高々1標的、各標的25〜27人、
  全736人というコホート構造を線形割当とsoft transportで反映します。
- AIA: 複数QI距離を背景データで密度補正し、既知の会員割合に合わせてchallenge行とC行を
  一対一で割り当てます。選択行には対応C行の`time`、非選択行には公開`A_bg`上で一般的な
  偶然的中が最も少ない範囲内の`time`を回答します。

## 練習データで実行

リポジトリ直下から次を実行します。

```powershell
powershell -ExecutionPolicy Bypass -File .\attack_phase\run.ps1 `
    -Dist .\participant_data `
    -OutputDir .\local-output\attack-phase-practice
```

出力:

- `F_mia.csv`: soft transport版（既定のMIA候補）
- `F_aia.csv`: 50%一意リンク版（既定のAIA候補）
- `variants/`: MIAのhard/unconstrained比較とAIAの選択割合比較
- `diagnostics.json`: 入力品質、割当制約、標的別の集計診断。個票は保存しません。

## 実戦データで実行

`B.csv`と`A_bg.csv`を含む自チームの参加者データ、および攻撃フェーズで配布された
`C_<k>.csv`、`aia_challenge_<k>.csv`、`mia_columns.json`の置き場を分けて指定できます。

```powershell
powershell -ExecutionPolicy Bypass -File .\attack_phase\run.ps1 `
    -Dist .\work\prelim\team24\participant_data `
    -Targets C:\path\to\prelim-attack-targets `
    -OutputDir .\local-output\attack-phase-prelim
```

このプログラムはトークンを読み書きせず、ZIP提出も行いません。実戦出力は、秘密トークンを
漏らさないローカル手順でZIP化し、必ず公式`pwscup2026.kit.validate`を通してから提出します。

### 予備戦 team24 の再現コマンド

配布一覧には自チームの`24`も含まれるため、計算前に`--exclude-targets 24`で除外します。
20対象のMIA総人数は、29対象で既知の736人を比例配分した事前値508を使います。この508は
CSV形式の必須条件や秘密真値ではなく、soft transport内部の事前仮定です。

```powershell
& .\starter\.venv\Scripts\python.exe .\attack_phase\generate_submission.py `
    --dist .\work\prelim\team24\runtime_dist `
    --targets .\work\prelim\team24\prelim-attack-targets `
    --exclude-targets 24 `
    --out-dir .\local-output\prelim\team24\attack-phase-20260827-v1 `
    --aia-active-fraction 0.5 `
    --aia-variant-fractions 0.25,0.5,0.75,1.0 `
    --mia-min-per-target 25 `
    --mia-max-per-target 27 `
    --mia-total-shared 508 `
    --mia-transport-epsilon 0.35
```

## 比較実験

- AIA: `-AiaActiveFraction 0.25`などで本命割合を変更できます。既定実行でも
  0.25/0.5/0.75/1.0の一意割当CSVに加え、C行の再利用を許す独立近傍版も生成します。
- MIA: `variants/F_mia_hard_capacity.csv`と`F_mia_unconstrained.csv`を比較できます。
- KDEが重い動作確認では`-SkipDomias`を使用できます。実戦候補ではDOMIAS込みを既定とします。

練習データには事務局だけが持つ正解ラベルがないため、ローカルで公式攻撃得点は計算できません。
練習結果は実装検証用であり、リーダーボード性能の推定には使いません。

## ルール準拠shadow評価

事務局限定の真値を推測せず、公開された人数構成と同じ既知の疑似コホートをローカルに作り、
キットの公式`score_mia_power` / `score_aia_power`を直接呼ぶ回帰評価器を用意しています。

```powershell
powershell -ExecutionPolicy Bypass -File .\attack_phase\evaluate.ps1
```

既定では2つのseedについて、無加工の正例、3%/10%摂動、転帰シャッフル、独立負例を評価し、
全MIA/AIAバリアントを比較します。MIAの29標的・1049候補・総重複736・各25〜27・候補の
標的一意性に加え、AIAの各群163件・発症16件・希少13件・希少かつ発症4件も再現します。
出力は`local-output/attack-shadow-evaluation/`です。

shadow用の`A_bg`は、攻撃器へ見せる参照群と、標的・対照を作る隠し群にattack非依存の固定hashで
分割します。これにより、疑似対照と同一の公開行が攻撃器にも見える評価リークを避けます。
公開データだけでは29標的の秘密コホートを一体として復元できないため、MIAとAIAは別々の
shadow標的群で評価します。AIA memberの希少ラベルは`B_self`真値、control側だけは公開
Mahalanobis検出器によるproxyです。

- `summary.json`: 本命方式のシナリオ別集計、バリアント順位、sanity check、解釈上の限界
- `scores.csv`: MIA×AIAの全組み合わせ得点
- `component_scores.csv`: MIA/AIA単体の得点
- `target_scores.csv`: 標的別得点
- `run_diagnostics.json`: 制約・transport収束・提出形式の検証結果
- `shadow_reference/`: 再現・監査用の疑似真値（公開練習データからのみ生成）

この評価は公式と同じ数式・標的平均を使いますが、疑似コホートと疑似防御を使うため、公式得点や
順位の予測ではありません。`independent`の非ゼロ値は有限標本下の負例ベースラインとして扱います。
