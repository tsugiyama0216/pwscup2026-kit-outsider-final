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

計算式とルール上の根拠は [RULEBOOK_BASIS.md](RULEBOOK_BASIS.md)、一覧は [MODEL_COMPARISON.md](MODEL_COMPARISON.md)、モデル別の特徴・攻撃内訳・結論は [MODEL_DIAGNOSTICS.md](MODEL_DIAGNOSTICS.md) を参照してください。各モデルの生データは `results/<model>/SUMMARY.md` と `resistance_report.json` に保存します。

## 新しいモデルの追加

`run.ps1` のモデル、C、scoreを指定すると、同じ攻撃・定数・20,000回偶然補正で評価し、比較表を再構築します。

```powershell
.\run.ps1 -Model M2 -C ..\M2\C.csv -Score ..\M2\score.json
```

評価基盤はCodaBenchへ送信せず、tokenも読みません。
