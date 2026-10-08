# ③ Gen-LRA固定カーネル版 + 局所time分布

このディレクトリは、計算式を追える独立した探索用攻撃です。入口は
`plugin.py:predict_world`。入力は評価器が渡す公開表だけで、回答ファイル、
他チームの正解、既存攻撃の予測を読みません。自チームの正解ラベルも使用しません。

## MIA

候補 x、公開合成表 S、背景参照表 R（m行）、S_k(x) を S 内の近傍 k 行とします。
7連続列 age/BMI/SBP/TG/HDL/ALT/FPG を用い、k=10（Sが小さければ全行）です。

1. Rの列中央値で欠損補完し、Rの平均と標本標準偏差で全表を標準化します。
   定数列の標準偏差は1とします。R全体が欠損の列、無限値、2行未満のRは拒否します。
2. 標準化Rの標本共分散を固有値分解し、固有値を最低1e-6にします。
   d=7として Silverman係数 h=(m(d+2)/4)^(-1/(d+4)) を掛け、
   Gaussian KDE の共分散を H=h^2 Cov_reg(R) に固定します。
3. 標準化Euclidean距離でSの10近傍を選びます。近傍距離の完全同順位は
   標準化7列の辞書順で決め、kを固定します。公開Sの行順には依存せず、乱数は
   使用しません。KDEの距離はHによるMahalanobis距離です。
4. p_R(s)=sum_{r in R} K_H(s-r)/m とし、xを加えた密度も同じHで計算します。

スコアの定義:

```text
p_R+x(s) = (m*p_R(s) + K_H(s-x)) / (m+1)
G(x) = sum_{s in S_k(x)} log(p_R+x(s) / p_R(s))
```

xの追加で近傍合成行の密度が増すほど、所属のスコアが上がります。
DOMIASの候補点における log p_S(x) - log p_R(x) と異なり、
候補の近くにある合成行を介して証拠を集計します。

数値計算には q=K_H(s-x)/(m*p_R(s)) を使います。
G=sum softplus(log q)-k*log(1+1/m) のうち後半は全候補共通なので、
順位が同じ `raw=log(sum softplus(log q))` を返します。
密度の下限置換やsigmoidは使わず、logsumexpで遠方の弱い信号を保持します。
log q < -30 では log(softplus(log q))=log q の漸近式を用います
（相対誤差は約4.7e-14以下）。真の同順位は維持します。
最後に共通 `score_to_confidence` で順位を[0,1]へ写します。
これは所属確率の校正ではありません。

## 論文・公開実装との違い

参照: [Gen-LRA論文v2](https://arxiv.org/html/2508.21146v2)、
[著者実装](https://github.com/joshward96/Synth-MIA/blob/main/synth_mia/attackers/gen_lra.py)。

論文の局所尤度比の式に基づき、固定カーネルの追加行密度を厳密に計算します。
著者の公開コードはRとR+xでSciPy KDEを再fitし、共分散と帯域を更新します。
本実装はHをRで固定する変種であり、その再fit実装の完全再現ではありません。
論文v2のk=10を採用します（公開コードの既定k=200ではありません）。
カテゴリ列はKDEへ入れません。この列制限と正則化も本探索の明示的設計です。

## AIA

`explore_common.core.local_time_predictions(release, challenge, background,
k=10, mode="window")` に委譲します。局所time分布から許容窓に入る件数を
最大化する予測で、詳細・既知属性・窓幅・境界処理は共通実装を参照してください。
モデル設定は `config.json` と一致させて固定しています。
戻り値の `diagnostics` に固定設定、参照行数、帯域係数、固定カーネルと
対数スコアの変種名を返し、評価器の `plugin_metadata.json` に保存されます。

## 検証

リポジトリ直下で:

```powershell
& .\main_round\attack\.venv\Scripts\python.exe -B -m unittest discover -s main_round/attack/exploration/explore_3_gen_lra/tests -v
```

テストは人工小データのみです。固定KDEの明示的追加密度との一致、遠方の
数値順位、特異共分散・欠損、リリースの複製、公開API出力を確認します。
予備戦データでの採点は統一評価側で行います。模擬得点は本戦の予測値ではありません。
