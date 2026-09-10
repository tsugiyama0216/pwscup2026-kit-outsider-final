# 根拠と確認記録

v0.6 · 2026-09-10 · 元のリポジトリ基準 `bc0e8c5` · 詳細編追加時の資料基準 `83a9846`

## この原稿で確認したこと

- 匿名化の提出済みCと、旧作業環境に残るv5出力のSHA-256が一致する。
- v5後処理レポートに記載された親C・出力Cのハッシュが実ファイルと一致する。
- 親Cとv5の共変量10列が、pandasで読み込んだ数値・カテゴリとして一致する。
- 匿名化の有用性数値は、保存済みの親C・v5の自己採点JSONから取得した。
- 最終MIAはv4とバイト単位で一致する。
- 最終AIAは対象ごとにコピー元と照合した。生成時と同じpandasの数値解釈で一致しており、CSVの末尾桁の表記まで一致するという意味ではない。
- 攻撃マニフェスト43件について、元ファイル・コピー先のサイズとSHA-256が一致する。
- 保存済み公式フィードバックから、AIA力と旧未得点7対象の寄与を再集計した。公式表示合計0.3407は、公式採点後に利用者が共有し、リポジトリへ保存した記録に基づく。今回の原稿作成でCodaBench画面を再取得したものではない。

機械的に照合した値は [verified_facts.json](verified_facts.json) に保存している。照合は既存出力を読み直して行った。DP生成の再実行、実入力Bの再配布、公式提出は行っていない。

本編と補足資料の架空の人数表・順位例・距離例・AIA的中率例は理解用である。実際の個票・実測値とは区別して、それぞれの説明に明記している。

## 根拠一覧

| 番号 | 出典 | 本編・補足で支える説明 |
|---|---|---|
| S1 | [予備戦攻撃時点のルールブック](../workspace/rulebook/PWS_Cup2026_ルールブック_prelim-attack-20260825.pdf) §3.1・§4.1・§4.2・§6.1〜6.3 | ブロック構造、MIA/AIA、人数、許容幅、有用性・攻撃力の定義 |
| S2 | [提出済み匿名化C](../anonymization/anonymization-submitted-v5/C.csv)、[匿名化提出物の案内](../anonymization/README.md) | 実際に提出したCの特定 |
| S3 | 下記「このPCの匿名化生成記録」 | 親Cの生成、A_bgによる転帰の付け替え、ρ、v5の設定、自己採点、選択過程の限界 |
| S4 | [匿名化生成器](../workspace/opendp_aim/scripts/generate_pwscup.py)、[転帰の付け替えの再適用](../workspace/opendp_aim/scripts/repostprocess_public_outcomes.py)、[設定の説明](../workspace/opendp_aim/config/README.md) | 事前に指定した9組のクロス集計＋AIM、区分番号への変換・提出用の値への変換、希少群の修復、リスク指標による距離 |
| S5 | [予備戦の攻撃ファイル集](../attack/README.md)、[最終生成器](../workspace/attack_phase_v9/generate_guarded_submission_v9.py) | 実際の20対象と提出物の構造 |
| S6 | [初期攻撃の説明](../workspace/attack_phase/README.md)、[初期攻撃生成器](../workspace/attack_phase/generate_submission.py) | 距離・DOMIAS・soft transport、508の事前仮定、一対一AIA |
| S7 | [v2の説明](../workspace/attack_phase_v2/README.md)、[v3の説明](../workspace/attack_phase_v3/README.md) | 追跡年数を観察上限へ近づける処理、11.0、確定負例163行、残り886行 |
| S8 | [v4の説明](../workspace/attack_phase_v4/README.md)、[v4 MIA](../workspace/attack_phase_v4/mia_private_conformal.py)、[v4生成器](../workspace/attack_phase_v4/generate_submission_v4.py) | 経験的較正、RRF、希少順位保全、AUCによる転帰判定 |
| S9 | [攻撃採点実装](../workspace/starter/src/pwscup2026/scoring/attack_score.py) | AIAの正規化超過率・希少側の標準誤差補正 |
| S10 | [independentの説明](../workspace/attack_phase_daybreak_independent/README.md)、[生成器](../workspace/attack_phase_daybreak_independent/generate_submission_daybreak_independent.py) | 六つの属性の組による対応、同じCの行の再利用、追跡年数の推定方法の切替 |
| S11 | [v6の説明](../workspace/attack_phase_v6/README.md)、[routed-guidedの結果](../attack/results/routed_guided_probe.json) | routed系の対象別構成と実測 |
| S12 | [v9の説明](../workspace/attack_phase_v9/README.md)、[最終生成器](../workspace/attack_phase_v9/generate_guarded_submission_v9.py) | 最終の対象別構成、未採用4方式 |
| S13 | [v7生成器](../workspace/attack_phase_v7/generate_submission_v7.py) | 三方式の共通部分の補集合、9.5・11.0 |
| S14 | [最終結果の説明](../attack/results/FINAL_RESULT.md)、[最終フィードバック](../attack/results/guarded_final.json) | 0.3407、MIA不変、AIA改善の分解と結果の由来 |
| S15 | [実際の親Cの生成設定](../../../pwscup2026-kit/work/prelim/team24/config/pwscup_aim_prelim_team24_v1.json)、[事務局から配布されたデータの仕様](../../../pwscup2026-kit/work/prelim/team24/runtime_dist/schema.json)、[生成設定作成コード](../../../pwscup2026-kit/opendp_aim/scripts/build_public_config.py) | 年齢区間、39組の内訳、事前に指定したクロス集計9組、A_bgから求めた区切り、親Cの生成パラメータ |
| S16 | このPCのOpenDP 0.15.1ソース `opendp/extras/mbi/` 内の `_table`・`_sequential`・`_fixed`・`_aim`・`_utilities` 各 `__init__.py`、[公式API説明](https://docs.opendp.org/en/stable/api/python/opendp.extras.mbi.html) | ノイズを加えた集計結果とモデルの引継ぎ、候補の展開・選択スコア・重要度、DP予算の配分、measure_split・rounds・サイズ制約 |
| S17 | Dwork・Roth, [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf#page=23), 命題2.1 | 後処理がDP保証を引き継ぐ条件。保存の前後が判断基準ではないこと |
| S18 | [routedの選別と年数の決定](../workspace/attack_phase_v5/aia_structured_ensemble.py)、[近傍密度による選別](../workspace/attack_phase_v6/aia_global_density.py)、[v6生成器](../workspace/attack_phase_v6/generate_submission_v6.py)、[保存済みv6診断記録](../workspace/local-output/attack-v6/diagnostics_v6.json) | 対象10の四つの重み・集団別の選択・イベント対応、対象20の第1・第3・第8近傍、Cの行の整理と実際の人数 |
| S19 | [予備戦キットの採点設定](../workspace/starter/src/pwscup2026/kit/kit_config.yaml) | low_fpr＝0.01、timeの許容幅0.5、希少AIAの標準誤差補正z＝1.64 |
| S20 | [事務局の有用性採点実装](../workspace/starter/src/pwscup2026/scoring/utility.py) | U_specのCox解析で使う9個の説明変数、追跡年数・発症の有無、BとCの解析の結論・予測性能の比較 |

S15の設定ファイルのSHA-256は `3a646111a693a659862805dbc7bceaf26f9486c7bae3832d7e9b0a3a699cdf80` で、親Cのprivacy reportの `config_sha256` と一致する。ファイル名は `v1` だが、内部のconfig_versionは `pwscup-aim-v4-prelim-team24-public-world-v1` である。AIM設定はこの一致したファイルを参照し、転帰の付け替えに使う重み0.31はv5のpostprocess reportを参照した。

S16のローカルソースは `pwscup2026-kit/opendp_aim/.venv/Lib/site-packages/opendp/extras/mbi/` にある。`rounds=16` は実測反復回数として記載していない。使用版では初期ステップ予算を決め、途中で予算を増やして終了する方式である。配分表は設定値からの計算であり、細かな浮動小数点調整は省いている。

AIM補足の改訂では、同じS15の設定とS16の候補展開処理を照合し、入力の二変数39組から、単変数11種類を含む計50種類の候補になることを確認した。県は39組に含まれないため、初期の頻度分布の対象12列と、候補に加わる単変数11種類は異なる。選択に使う重要度は設定のweightそのものではなく、変数の重なりも使って計算され、今回の「年齢×転帰」は13.6、「年齢×BMI」は7.7となる。これらは設定と使用版の処理から確認した値であり、実際に選ばれた項目・順序・回数を示すものではない。匿名化の生成は再実行していない。

選択スコアの説明は、今回使用したOpenDP 0.15.1に合わせた。原論文では測定ノイズの期待誤差を減点として説明しているが、使用版の実装は選択用ノイズのスケールとマス数から減点を計算する。そのため補足では「ノイズの規模とマス数に応じた減点」と表現し、原論文の測定ノイズの式を今回の実装の式として転記していない。

第1部の全体の流れと各ステップの処理の分担は、S4・S16と保存記録を照合して確認した。v5は保存済みの親Cに転帰の付け替えを再適用したもので、AIMの再実行ではない。今回の確認ではDP生成を再実行していない。

### ノイズ・予算と本文の構成の確認

第1部は図と本文の番号が対応する10ステップへ再構成した。S15の実際の生成設定とS4・S16を再確認し、頻度分布を作るステップ3にノイズと予算、ステップ5に集計の選択と反復内の配分、ステップ9・10に転帰の付け替えをまとめた。

人数へのノイズは、使用したOpenDP 0.15.1の整数の集計処理と、[ガウスノイズの公式実装](https://docs.rs/opendp/latest/opendp/measurements/fn.make_gaussian.html)を照合した。初期の頻度分布は整数の人数を扱うOpenDPの集計処理、9組のクロス集計とAIMの集計は `_utilities.make_noise_marginal` で整数に変換してからノイズを加える処理である。今回のρによる予算管理では、離散ガウス分布のノイズを使う。初期12列のノイズの大きさ約13.1は、1人1行、総予算0.1、初期配分35％という設定に対する値であり、実際に加えた乱数や、出力Cの誤差の実測値ではない。

集計の選択は `_aim` の `then_noisy_max` と、[OpenDP 0.15.1の選択処理](https://docs.rs/opendp/latest/src/opendp/measurements/noisy_top_k/mod.rs.html)を照合した。今回のρによる予算管理では、ガンベル分布のノイズを加えて最大の候補を選ぶ方法に相当する。人数に加える離散ガウス分布のノイズと区別して説明した。

総予算と段階ごとの配分は自チームの設定に由来する。OpenDPによる配分と反復内の調整は、S16の `_table`・`_sequential`・`_aim` で確認した。総予算0.1、初期35％、残り3対1が最適であると示した比較検証の記録は、今回確認した資料にはない。新しい匿名化データの生成や公式提出は行っていない。

### このPCの匿名化生成記録（S3）

最新リポジトリの一般的なREADMEだけでは、実際の提出v5の生成経路を確定できなかった。そこで、保全した旧環境の次のファイルを読み、提出Cとの対応を確認した。相対リンクはこのPCのフォルダー構成を前提としている。

- [親Cのprivacy report](../../../pwscup2026-kit/local-output/prelim/team24/opendp_aim/release-001/privacy_report.json)
- [v5のpostprocess report](../../../pwscup2026-kit/local-output/prelim/team24/opendp_aim/postprocess-v5-uspec-001/postprocess_report.json)
- [親Cのローカル自己採点](../../../pwscup2026-kit/local-output/prelim/team24/opendp_aim/release-001/score.json)
- [v5のローカル自己採点](../../../pwscup2026-kit/local-output/prelim/team24/opendp_aim/postprocess-v5-uspec-001/score.json)
- [v5の選択過程の監査](../../../pwscup2026-kit/work/prelim/team24/decisions/uspec-v5-selection-audit.json)

提出CのSHA-256：`efd38bf8ab973767caad34ce2b0ef9991b1e461bd7c1f9e17da06b9a847aaa14`

上記ローカル記録の存在を確認したうえで、匿名化v5を説明している。これらは別PCやGitHub上で自動的に参照できるとは限らない。記録の全文や生のBは原稿へコピーしていない。

## 一般的な用語・アルゴリズムの出典

- [総務省統計局「クロス集計」](https://www.stat.go.jp/dstart/point/seminar/02/3-2-2.html)。二変数の区分を組み合わせて人数を数える処理をクロス集計、結果の表をクロス集計表・分割表と呼ぶことを確認した。今回の資料では「事前に指定した9組のクロス集計表を作成する」と表現する。9組の指定とノイズの付加はS4・S15・S16で確認した。
- McKenna, Mullins, Sheldon, Miklau, [AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data](https://arxiv.org/abs/2201.12677)。AIMの選択・測定・更新の考え方を参照。
- [OpenDP: Synthetic Data](https://docs.opendp.org/en/stable/getting-started/statistical-modeling/synthetic-data.html)。DP集計から合成データを作る後処理の説明を参照。
- [OpenDP 0.14 announcement](https://opendp.org/2025/09/12/announcing-opendp-library-0-14/)。低次元周辺集計からPrivate-PGMで同時分布を推定する役割を参照。

外部資料は一般原理の説明に使用し、自チームの実測結果の根拠には使用していない。実行当時の親Cの生成記録はOpenDP 0.15.1・mbi 1.1.0である。

## 読み違えやすい研究履歴

旧README・引き継ぎには、当時の「残り1回」「未提出」などが残っている。最終結果はv9提出による0.3407であり、予備戦の提出枠は残っていない。

v7の旧シャドウ得点換算は、本番予測として撤回されている。この原稿では将来得点の予測に使っていない。v9の新しい4方式は不採用である。

旧Word教材は初期v1の資料なので、その本文を最終v9の説明として転載していない。

## 本文と補足の範囲

本編では、何を入力し、どの処理を行い、何を提出したかを説明する。AIMの役割と主な設定、A_bgを用いた後処理、最終攻撃の構成と結果を含む。

[匿名化の補足資料](SUPPLEMENT_AIM.md)には、AIMの集計選択・分布推定、全パラメータと予算配分、AIMの外で行ったドナー選択の距離計算を収録する。[攻撃の補足資料](SUPPLEMENT_ATTACK.md)には、MIAのスコア・順位の計算、AIAの採点式・判定条件、v7の集合による表記、最終提出に採用しなかった検討を収録する。

[攻撃のステップごとの詳細資料](ATTACK_STEP_BY_STEP.md)には、最終提出につながった各方式を、指標の作成から回答まで順に収録する。本編・補足・詳細編は同じ根拠番号を使う。DPの一般説明とコードの解説は含めない。

## 2026-09-10：攻撃の詳細編で追加確認したこと

- v3側のMIAが、v2で試した近傍密度の混合ではなく、主方式に残したv1のtransportと確定負例の降格を使うこと。
- v4の四つのview、距離と双方向の比の較正、0.55・0.45・0.20・0.25の重み、相手間の差の重み0.50、RRFの定数10。
- v4のAIAが、163人の選別と一対一の対応を同時に決めること。対応が遠い場合の年数は、中間値への補間ではなく10.0への切替であること。
- independentの六つのview、選別の重み、A_bgの近傍48行による基準、残差補正の重み0.25、AUCの切替基準0.60。
- 対象10・20で採用したroutedの方式、集団別の選択人数、Cの整理前後の行数。
- 最終F_aia.csvで、対象6・7・8・9・17・18・19の9.5回答数が、それぞれ217・253・211・274・223・236・268であること。
- 確定負例との比較から新しいMIAスコアまでの通し計算、RRF、一対一の割当、AUC、残差補正を、架空の値で確認したこと。

最終MIAがv4と同一であることと、前回の提出からファイルが不変であることは区別する。ここでは前者の一致と、前回との採点値の一致を述べる。

一般的な指標の説明には、[DOMIAS原論文](https://proceedings.mlr.press/v206/breugel23a.html)、[RRF原論文](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf)、[scikit-learnのAUC説明](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)を確認した。自チームの設定値・採用経路・結果の根拠は、保存された実装と記録である。

詳細編の数値例は架空例であり、実際の個票を復元・転載したものではない。攻撃や公式提出を再実行してはいない。
