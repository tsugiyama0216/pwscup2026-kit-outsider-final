# 攻撃バージョン5

v5は、実測v4を基準に残り3提出を情報効率よく使うための後処理レイヤーです。元の実
`B.csv` / `A_bg.csv` が手元にない状態でも、commit済みのv3/v4/独立候補と公開攻撃標的だけで
再生成できます。候補生成はネットワーク接続、token読込、CodaBench提出を行いません。

## 実測基準

| 候補 | 得点 | MIA all | MIA rare | AIA all | AIA rare |
|---|---:|---:|---:|---:|---:|
| v4 | 0.292 | 0.1024 | 0.3333 | 0.1171 | 0.1000 |
| 現在1位 | 0.368 | 0.1039 | 0.2500 | 0.1827 | 0.0519 |

v4はv2からMIA allだけが改善し、AIA二内訳は変化しませんでした。このためv5はAIA探索を
優先し、MIAはv4の低FPR上位だけを小さく修正します。

## 生成される候補

- `attack-candidate-v5-probe-independent`: v4 MIAをbyte-for-byte固定し、AIAだけを独立方式へ変更
- `attack-candidate-v5-structured`: v4 MIAを固定し、公開matching構造を使うAIAへ変更
- `attack-candidate-v5-tail12`: structured AIAに、MIA top-12一意化を組み合わせる

各candidateディレクトリには`F_mia.csv`と`F_aia.csv`だけを書きます。診断と比較variantは
`local-output/attack-v5/`へ分離します。

## AIAの変更

公開ルール上、challengeは163 memberと163 matched controlです。実ファイルを調べると、18/20
targetでは`(sex, prefecture, 5歳帯)`の各層がすべて偶数で、各層のmember数をちょうど半数に
できます。target 10と23の奇数層だけは、全体が163になるよう次点scoreの高い層へceil枠を
配ります。

本命`consensus_v4_lean__independent_fallback`は次の順で作ります。

1. v4 active、独立方式active、公開Cへのmultiview近接度を融合
2. 公開matching層ごとにhalf quotaを適用
3. 独立方式の時刻を優先し、無い場合はv4、両方無い37行だけ10年を回答
4. 他target Cにも完全一致する行と、C内の重複行を近接度計算から除外

年齢順の隣接2行から1行を選ぶ追加制約、全activeを10年にするdecoder、固定event件数capは
shadowで悪化または不安定だったため採用していません。

## MIAの変更

v4はMIA allを実測で改善しましたが、top-12の240枠に9個のcross-target重複があります。候補は
高々1targetのmemberなので、Hungarian assignmentで9枠だけを一意化します。

- private 163行は全targetでbottomのまま
- eligible rare 11行のtarget内相対順はv3と完全一致
- top-12重複は9から0
- 全列tie-free、scoreは有限な`(0,1)`

## 実行

```powershell
.\starter\.venv\Scripts\python.exe -m attack_phase_v5.generate_submission_v5
.\starter\.venv\Scripts\python.exe -m unittest discover -s attack_phase_v5/tests -v
```

提出ZIPは候補生成と分離したpackagerで作ります。packagerだけがteam tokenを読み、CSVの
SHA-256一致と公式2層validator合格を確認してから出力ZIPを更新します。token内容は表示しません。

```powershell
.\starter\.venv\Scripts\python.exe -m attack_phase_v5.package_submission_v5 `
    --candidate-dir .\submissions\prelim\team24\attack-candidate-v5-probe-independent `
    --out .\local-output\attack-v5\attack-v5-probe-independent.zip
```

最初は`attack-candidate-v5-probe-independent`だけを提出し、20target分の`aia_per_target`を保存します。
2回目に`attack-candidate-v5-tail12`を試し、3回目はv4・probe・structuredのtarget別勝者をblock単位で
合成するために残します。CodaBenchの詳細結果を保存できない場合は、2回目をMIA固定の
`attack-candidate-v5-structured`へ切り替えます。

詳細結果をJSONへ保存した後は、`blend_feedback_winners.py`へ候補CSVと同じlabelの得点JSONを
渡すと、各targetの公式`aia_per_target`最大候補から326行全体をコピーします。JSONは
`{"aia_per_target": {"1": 0.1, ...}}`、`_breakdown`配下、またはtarget→scoreの直接mapを受け付けます。
同点時は先に指定した候補を維持します。生成後は論理validatorを通し、同じpackagerで最終ZIPにします。
