# team10 共通の攻撃耐性評価

S1～S2、R1～R9、M1以降を同じ条件で比較するためのローカル耐性評価基盤です。有用性だけで候補を選ばず、MIA・AIAの攻撃信号と並べて判断します。評価結果だけをM配下に保存し、S・Rの成果物は変更しません。

## 使用する攻撃

| 種類 | 実装 | 位置づけ |
|---|---|---|
| MIA | 運営 `overlap_mia`（DOMIAS KDE密度比） | 本戦キットの参照攻撃 |
| MIA | `team_v4_v9_multiview` | 予備戦v4/v9の多視点近傍・希少側重視をローカル候補へ移植 |
| AIA | 運営 `aia_predict` | sex・prefectureブロック内QI最近傍 |
| AIA | `team_multiview_linkage` | 予備戦のQI照合を3視点で行い、予測timeの中央値を採用 |
| AIA | 10年固定 | データを見ないsanity baseline。最大リスクには含めない |

正式な本戦攻撃challengeはまだこのリポジトリにないため、B全行をmember/標的、公開A_bgをnonmember/対照とする代理評価です。結果はモデル間比較用で、公式protection・Anon・リーダーボード予測ではありません。

各指標の目的・計算・攻撃実装・解釈は [METRICS_GUIDE.md](METRICS_GUIDE.md)、ルール上の根拠は [RULEBOOK_BASIS.md](RULEBOOK_BASIS.md)、一覧は [MODEL_COMPARISON.md](MODEL_COMPARISON.md)、モデル横断の特徴は [MODEL_DIAGNOSTICS.md](MODEL_DIAGNOSTICS.md) を参照してください。各モデルには `results/<model>/DETAILS.md`、`SUMMARY.md`、`resistance_report.json` を保存します。

## 新しいモデルの追加

`run.ps1` のモデル、C、scoreを指定すると、同じ攻撃・定数・20,000回偶然補正で評価し、比較表を再構築します。

```powershell
.\run.ps1 -Model M2 -C ..\M2\C.csv -Score ..\M2\score.json
```

評価基盤はCodaBenchへ送信せず、tokenも読みません。

## 加工データの防御スコアを一度に出す

任意の加工済み `C.csv` と、その候補の有用性スコアJSONがあれば、攻撃力代理、Anon（防御）代理、総合点代理を1つのJSONとMarkdownにまとめられます。

```powershell
.\run_defense.ps1 `
  -Model R12 `
  -C ..\R\R12\C.csv `
  -Score ..\R\R12\report.json `
  -Phi 0.026887
```

team10本戦については、提供された `φ_i=0.026887` を `phi_team10.json` に保存しており、`run_defense.ps1` が自動適用します。別の値で再計算する場合だけ `-Phi <値>` を指定してください。ルールブック §6.2.2 のとおり、AIAは `clip(R - φ_i, 0, 1)` に補正してからMIAと比較します。

計算方法をローカルで再現したい場合は、正解列付きのAIA challenge（`is_target`, `time`, `onset` を含む）を用意して次を実行します。

```powershell
# リポジトリルートから実行
& .\prelim\workspace\starter\.venv\Scripts\python.exe `
  .\main_round\anonymization\team10\M\evaluation\scripts\estimate_phi.py `
  --challenge .\challenge_with_truth.csv `
  --data .\main_round\anonymization\team10\participant_data `
  --kit-root .\prelim\workspace `
  --out .\phi_estimate.json
```

この推定器はルールブックの「A_bgから1,049行を置き換え、同じAIA採点を1,000回繰り返して平均する」手順を実装しています。公式の正解付きchallengeと事務局のサンプリング手順がない段階では、出力は公式φ_iではありません。

すでに `resistance_report.json` がある場合は、攻撃を再実行せずφ_iだけを反映できます。

```powershell
.\run_defense.ps1 -Model R12 -C ..\R\R12\C.csv -Score ..\R\R12\report.json `
  -Phi 0.026887 -ResistanceReport ..\R\R12\attack_evaluation\resistance_report.json
```

出力先は `results/<Model>/defense_score.json` と `DEFENSE_SCORE.md` です。主な値は次のとおりです。

- `attack_power_proxy = max(MIAリスク, AIAリスク)`
- `anon_proxy = 1 - attack_power_proxy`
- `overall_score_proxy = min(U, anon_proxy)`

運営の本番攻撃候補と最終的な公式Anonは攻撃フェーズ後にしか分かりません。φ_iの計算方法はルールブックに定められていますが、計算には対象コホートのAIA標的・対照構成と、A_bgから1,049行を置き換える1,000回の実測が必要です。したがって、φ_i未入力時はAIA控除前、入力時は控除後のローカル代理値を出します。MIAの正規化前βも併記し、ローカル候補集合による飽和の影響を確認できるようにしています。
