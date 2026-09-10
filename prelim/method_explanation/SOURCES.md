# 根拠と確認記録

v0.5 · 2026-09-09 · 元のリポジトリ基準 `bc0e8c5`

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
| S4 | [匿名化生成器](../workspace/opendp_aim/scripts/generate_pwscup.py)、[転帰の付け替えの再適用](../workspace/opendp_aim/scripts/repostprocess_public_outcomes.py)、[設定の説明](../workspace/opendp_aim/config/README.md) | 固定集計＋AIM、区分番号への変換・提出用の値への変換、希少群の修復、リスク指標による距離 |
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
| S15 | [実際の親Cの生成設定](../../../pwscup2026-kit/work/prelim/team24/config/pwscup_aim_prelim_team24_v1.json)、[事務局から配布されたデータの仕様](../../../pwscup2026-kit/work/prelim/team24/runtime_dist/schema.json)、[生成設定作成コード](../../../pwscup2026-kit/opendp_aim/scripts/build_public_config.py) | 年齢区間、39組の内訳、固定9組、A_bgから求めた区切り、親Cの生成パラメータ |
| S16 | このPCのOpenDP 0.15.1ソース `opendp/extras/mbi/` 内の `_table`・`_sequential`・`_aim`・`_utilities` 各 `__init__.py`、[公式API説明](https://docs.opendp.org/en/stable/api/python/opendp.extras.mbi.html) | 一列集計の予算配分、3対1の配分、各ステップのmeasure_split、roundsの扱い、サイズ制約 |
| S17 | Dwork・Roth, [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf#page=23), 命題2.1 | 後処理がDP保証を引き継ぐ条件。保存の前後が判断基準ではないこと |

S15の設定ファイルのSHA-256は `3a646111a693a659862805dbc7bceaf26f9486c7bae3832d7e9b0a3a699cdf80` で、親Cのprivacy reportの `config_sha256` と一致する。ファイル名は `v1` だが、内部のconfig_versionは `pwscup-aim-v4-prelim-team24-public-world-v1` である。AIM設定はこの一致したファイルを参照し、転帰の付け替えに使う重み0.31はv5のpostprocess reportを参照した。

S16のローカルソースは `pwscup2026-kit/opendp_aim/.venv/Lib/site-packages/opendp/extras/mbi/` にある。`rounds=16` は実測反復回数として記載していない。使用版では初期ステップ予算を決め、途中で予算を増やして終了する方式である。配分表は設定値からの計算であり、細かな浮動小数点調整は省いている。

第1部第2節の処理の分担は、S4・S16と保存記録を照合して確認した。v5は保存済みの親Cに転帰の付け替えを再適用したもので、AIMの再実行ではない。今回の確認ではDP生成を再実行していない。

### このPCの匿名化生成記録（S3）

最新リポジトリの一般的なREADMEだけでは、実際の提出v5の生成経路を確定できなかった。そこで、保全した旧環境の次のファイルを読み、提出Cとの対応を確認した。相対リンクはこのPCのフォルダー構成を前提としている。

- [親Cのprivacy report](../../../pwscup2026-kit/local-output/prelim/team24/opendp_aim/release-001/privacy_report.json)
- [v5のpostprocess report](../../../pwscup2026-kit/local-output/prelim/team24/opendp_aim/postprocess-v5-uspec-001/postprocess_report.json)
- [親Cのローカル自己採点](../../../pwscup2026-kit/local-output/prelim/team24/opendp_aim/release-001/score.json)
- [v5のローカル自己採点](../../../pwscup2026-kit/local-output/prelim/team24/opendp_aim/postprocess-v5-uspec-001/score.json)
- [v5の選択過程の監査](../../../pwscup2026-kit/work/prelim/team24/decisions/uspec-v5-selection-audit.json)

提出CのSHA-256：`efd38bf8ab973767caad34ce2b0ef9991b1e461bd7c1f9e17da06b9a847aaa14`

上記ローカル記録の存在を確認したうえで、匿名化v5を説明している。これらは別PCやGitHub上で自動的に参照できるとは限らない。記録の全文や生のBは原稿へコピーしていない。

## 一般的なアルゴリズムの出典

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

本編と補足は同じ根拠番号を使う。DPの一般説明とコードの解説は含めない。

今後さらに詳しい例を追加する場合は、次の内容を扱える。

- 一つの合成行について、区分番号から値への変換とドナー選択を追う例。
- 一つのMIA候補について、距離の較正・RRF・希少候補内の順位保持を追う例。
- 一つのAIA対象について、候補の選別と追跡年数の推定を分けて示す図。

これらは今後の追補候補であり、現在の本文に実際の個票を追跡する計算を収録したという意味ではない。
