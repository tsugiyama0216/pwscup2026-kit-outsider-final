# Rv6–Rv11：新採点に渡す6候補

2026-10-09の依頼に基づき、Rv2を基準に、異なる仮説を比較できる6版を作成しました。保存先は今回指定された `main_round/anonymization/team10/attack/attackR/` です。既存Rv2/Rv4/Rv5、M/Tのソース・予測・検証記録は変更していません。

**6版の公式得点・新採点の得点は未測定です。「改善を狙った候補」であり、Rv2を上回ったモデルとしては扱いません。** 杉山採点、retest、mock_TSの得点による選択・調整は行っていません。

| 版 | MIA | AIA | 比較で確かめる仮説 |
|---|---|---|---|
| Rv6 | 複数近傍の局所密度、2属性ビュー | Rv2を保存 | 最近傍1件への依存を下げ、低FPR側の順位を改善できるか |
| Rv7 | 共分散で補正した距離、全連続属性とQI | Rv2を保存 | 相関する検査値の重複を抑え、加工時刻の影響を避けられるか |
| Rv8 | 人物ごとの対象共通傾向を弱く補正 | Rv2を保存 | どの対象にも高く出る一般的な似やすさを減らせるか |
| Rv9 | Rv2を保存 | 近傍の±0.5年窓と背景との差 | 平均時刻を外す多峰性への対処が有効か |
| Rv10 | Rv2を保存 | ExtraTreesの葉から条件付き時刻分布 | 非線形な属性と時刻の関係を利用できるか |
| Rv11 | Rv6/Rv7/Rv8の固定重み統合 | 近傍・森林の双方が支持する候補 | 異なる方式の一致を使って変更の不確実性を抑えられるか |

## Rv2の弱点と設計

ユーザー提供の公式結果は `official_feedback_Rv2_user_supplied.json` に原値を保存します。MIAのAUCは0.6213502640ですが、FPR=0.01でのTPRは0.0744755245です。AIAの全体p_m=0.5716675962とp_c=0.5270496375の差は約0.04462、希少層は0.3811188811と0.3986013986で差が負です。一般的によく当たる時刻予測だけではメンバー特有の情報を増やせない、と解釈しました。

これは集計値からの仮説です。公式の個人別正解や隠された加工方法はありません。対象9/11だけに特別なパラメーターを設けていません。対象別MIAの1〜4は人数ではなくTPR帯です。

Rv6は背景に対するCの距離比を1/3/10近傍で集約します。7連続属性と、時刻を加えた8属性を別々に測定し、背景分布内の順位を使ってRv2の順位と混合します。Rv7は背景共分散を縮小推定して白色化し、7連続属性と5つの連続QIを使います。背景の自己参照では当該行を除きます。

背景分布内の順位は**本番のFPR保証でも、メンバー確率でもありません**。Cと背景の標本数差は同じ構成の背景参照と比較しますが、背景と候補集団のずれは残ります。Rv6/Rv7はRv5より広い順位変更を意図した別仮説です。

Rv8は各対象の順位をロジット化し、人物ごとの対象間中央値を弱く差し引きます。同じ人物が複数対象に含まれる可能性を維持し、排他的な対象割当や陽性人数固定は行いません。真の共有メンバーに共通する信号まで抑える可能性があり、独立した候補として検証します。

Rv9/Rv10は候補時刻の±0.5年に入るCの重みと背景の重みの差を使います。Rv10は96本のExtraTreesで条件付き葉分布を作り、森林の平均回帰値をそのまま提出しません。学習ラベルは公開された加工Cと背景のtimeであり、公式AIAの正解ではありません。C時刻自体の加工、Cと背景の学習標本数差、森林の過適合は未解決のリスクです。

Rv11はMIAを0.4/0.3/0.3で統合し、AIAはRv2・Rv9・Rv10の出力候補から選びます。近傍と森林のどちらも背景差を悪化させない候補だけ変更します。離れた時刻モードを数値平均しません。統合版が単独版より高得点とはまだ言えません。

## 凍結・検証

- `protocol.json`：6版共通の固定パラメーター。全対象同じ値です。
- `dependencies.json`：本番予測前の推論コード・入力・Rv2予測CSVのハッシュ固定。
- `preservation.json`：作業開始時点の過去R成果物のハッシュ。ローカル保全記録であり、新環境の推論に必要な依存一覧は `dependencies.json` です。
- `test_model.py`：9件の数値・並べ替え・入力非変更・時刻分布・既知陰性テスト。採点用模擬問題ではありません。
- 各版の `verification.json`：2つの独立プロセスで生成したCSVの一致、CSV再読込、実team10のIDと公式ZIP形式、8種類の不正提出の拒否。
- `production_summary.json`：Rv2からの順位・上位30件・時刻変更数。得点差ではありません。

公的配布情報から厳密に導ける163件の既知非共有行は0に保ちます。この163件だけを用いたFPR推定は行いません。

## 保存・再生成

各 `RvN/submission/` に `F_mia.csv`、`F_aia.csv`、PC用の `team10_RvN_attack.zip` を保存します。MIAは1,049行×22対象、AIAは7,172回答です。GitHubにはコード・2つのCSV・検証記録を保存します。認証情報を含むZIPはPC上でGit除外とします。CodaBenchへの提出は行いません。

既存の `main_round/attack/.venv` を利用します。新しいPCでは `main_round/attack/setup.ps1` を実行してください。保存済み予測を再生成する場合、未作成の出力フォルダーを指定します。

```powershell
./main_round/anonymization/team10/attack/attackR/Rv6/run.ps1 -OutputDirectory ./main_round/anonymization/team10/attack/attackR/Rv6/reproduced
```

共通処理の検証と一括生成は次の通りです。1版指定でも整合性のため6版の共通計算を実行します。`package.py` は既存提出物の上書きを拒否します。

```powershell
./main_round/attack/.venv/Scripts/python.exe -B ./main_round/anonymization/team10/attack/attackR/rv6_series/test_model.py
./main_round/attack/.venv/Scripts/python.exe -B ./main_round/anonymization/team10/attack/attackR/rv6_series/infer.py --output ./main_round/anonymization/team10/attack/attackR/rv6_series/.work/reproduced_all
```

新しい採点システムへの接続は `EVALUATOR_HANDOFF.md`、未実装の追加案は `RESEARCH_BACKLOG.md` を参照してください。

GitHubから取得した別PCで、保存済みCSVから提出ZIPを組み立てる場合は次を使います。依存・CSVハッシュと実IDを検査し、そのPCにある配布原本tokenを組み込みます。既存の正しいZIPはそのまま検証し、内容が異なるZIPは上書きしません。ZIPの日時属性が変わっても、提出CSVの中身は保存版と一致させます。

```powershell
./main_round/attack/.venv/Scripts/python.exe -B ./main_round/anonymization/team10/attack/attackR/rv6_series/assemble_saved_zip.py --model Rv6
```
