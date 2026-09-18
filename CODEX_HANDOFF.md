# PWSCUP2026 引き継ぎ

**2026-09-18、ユーザー提供のR6公式結果から[R7](main_round/anonymization/team10/R/R7/README.md)を作成。** 多数派の条件付きコピュラ補正0.5、外れ具合の分布補正0.5、半径を保持する希少群コピュラ補正0.5を施し、集計Coxの有意性を保つ目標で合成発症・打切りの組を4回交換。90設定（範囲不適合・上限打切り含む）を9段階で比較し、U=0.9568840160028896（gen=.965034580679937、spec=.9568840160028896、rare=.9639190249630962、valid=1）を採用。検出希少57。独立ホールドアウト評価ではなく、公式0.95超え・攻撃耐性は未保証。12テスト合格、最終成果物検証はR7/verification.json。R1〜R6を保持。ユーザーはこのPCと既存非公開GitHubへの保存を指示しており、`main_round/anonymization/team10/R/R7/submission/team10_R7_defense.zip`を含む。CodaBenchへは送信しない。

**2026-09-18、R4公式結果を受けてコピュラによる新しい[R6](main_round/anonymization/team10/R/R6/README.md)を作成。** 以前のR5由来R6はユーザーが中止・削除しており、復元していない。新R6は固定R4の検出希少55行に、B検出71行の周辺分布と潜在共分散に基づくガウシアン・コピュラ変換を50%適用する。追加Cox補正なし。12候補を比較し、U=0.9193239002532438（gen=.9373477726479545、spec=.9278269436619543、rare=.9193239002532438、valid=1）を採用。全4指標はR4公式と同等以上。10人工データテストと公式提出形式検証成功。検証記録はR6/verification.json。R4の公式MIA=0、AUC=.5176171875000001はR6へ引き継がず、R6の公式攻撃耐性未評価。ユーザーはこのPCとGitHubへの保存を指定し、`main_round/anonymization/team10/R/R6/submission/team10_R6_defense.zip`を含む。既存非公開originのみに保存し、R1〜R5を保持、CodaBenchへは送信しない。

**2026-09-18、ユーザー指定の逐次条件付き合成で[R5](main_round/anonymization/team10/R/R5/README.md)を作成。** CARTの各列条件分布を順次サンプリングし、競合リスクCoxによる事象生成と強度0.5の集計補正を組み合わせる。葉サイズ20/40/60×補正0/0.5の6候補を比較し、最小葉20を採用。ローカルU=0.8079092630783484でR4より低い。1,049行・13列、形式検証成功。R1〜R4は不変。ユーザーはこのPCと既存非公開GitHubに提出データをR5として格納するよう依頼しており、`main_round/anonymization/team10/R/R5/submission/team10_R5_defense.zip`を含む。公式攻撃耐性未評価、差分プライバシー保証なし、CodaBenchへは送信しない。検証結果はR5/verification.json。

予備戦は **v9 guarded、確定攻撃力0.3406681446132233（4位）** で終了しました。過去の0.3407は攻撃力の表示丸めです。**総合得点は0.4223063973063974（17位）**。提出回数は残っていません。
9月9日の[公開結果・原本・照合記録](prelim/results/20260909/README.md)を追加しました。公開AIAの攻撃者4→標的23の欠落と平均分母の要確認点も記録しています。team24の公開攻撃CSVは保存済み最終v9と完全一致しています。

本戦版[main-process-20260912のPDFと変更点](main_round/rulebook/README.md)は保存済みです。**2026-09-16に、本戦・加工フェーズのteam10配布データを[main_round/anonymization/team10/](main_round/anonymization/team10/README.md)へ配置しました。** 予備戦のteam24とはコホート番号が異なります。本戦公式キットは固定コミットで導入・検証済みです。

**2026-09-17に[S2](main_round/anonymization/team10/S/S2/README.md)を作成しました。** S1の公式返却値はローカル有用性U=0と一致し、原因は列間相関の崩れでした。mbi 1.1.0の条件付きroundをsampleへ切り替えたS2はローカルU=0.5484134662543412。同じ保存済みDPモデルでもround=0、sample=0.5252213432784834となり修正効果を確認しています。S1は凍結保存し、S2は1回生成、29テスト・公式形式検証成功、提出ZIP作成済み。S2のCodaBench提出はユーザーが行います。ユーザーは、この状態の非公開GitHubへのアップロードとmainへの反映を許可しました。

- [リポジトリ構造と旧パス対応](REPOSITORY_LAYOUT.md)
- [予備戦の全体案内](prelim/README.md)
- [最終攻撃CSVと公式結果](prelim/attack/README.md)
- [匿名化フェーズの提出物](prelim/anonymization/README.md)
- [詳細な研究引き継ぎ](prelim/workspace/CODEX_HANDOFF.md)
- [最終実測結果の説明](prelim/workspace/attack_phase_v9/FINAL_RESULT.md)
- [本戦の格納先](main_round/README.md)
- [本戦・加工フェーズのteam10データ](main_round/anonymization/team10/README.md)

**2026-09-17に[R1](main_round/anonymization/team10/R/R1/README.md)を作成しました。** 性別・喫煙別ガウシアン・コピュラと競合リスクCoxによる1,049行・13列の完全合成です。S2のコード・モデル・生成データは不使用。ローカル有用性U=0.564209862140604、公式形式検証と15テスト成功、同一乱数種のCSVバイト再現も確認済みです。匿名性・差分プライバシーの保証はありません。ユーザーはR1一式の非公開GitHubへのアップロードを依頼し、トークン入り `main_round/anonymization/team10/R/R1/submission/team10_R1_defense.zip` を含めることも明示的に許可しました。この追加許可はR1の同ZIPだけに適用し、CodaBench提出は行いません。

**2026-09-17に作成したR2を、ユーザー指定の[R2](main_round/anonymization/team10/R/R2/README.md)へ格納しました。** R1と同じ階層に提出CSV・ZIP・実装・評価記録をまとめています。モデル名とZIP名はR2のまま、CSV・ZIP・生成コード・評価レポートのバイトを保存しています。ローカル有用性U=0.7693769302515455、49テストと生成時の公式形式検証成功。ユーザーが追加の再生成確認を取り消したため、生成後のReproduceは未実行です。ユーザーは提出済みR2の非公開GitHubへの保存を依頼しており、この許可は`main_round/anonymization/team10/R/R2/submission/team10_R2_defense.zip`を含みます。こちらからCodaBenchへの送信は行いません。

**2026-09-17、ユーザー提供のR2公式結果を受けて[R3](main_round/anonymization/team10/R/R3/README.md)を作成しました。** R2の保存済み合成CSVへ、集計Cox係数と共分散に基づく強度0.5の部分補正を適用します。全9係数で有意・非有意が元Bと一致。6候補を比較し、ローカルU=0.8661244917109793（U_gen=0.9187884920473647、U_spec=0.9116907071412028、U_rare=0.8661244917109793、U_valid=1）を採用しました。一般有用性はR2から微減します。R2の参照MIA=0はR3へ引き継げず、R3の攻撃耐性は未評価です。17テスト成功、提出ZIP作成。生成時の公式形式検証・再現確認の記録はR3/verification.jsonを参照してください。R1・R2は変更せず、R2の再生成も行いません。作成後の追加依頼により、R3一式と`main_round/anonymization/team10/R/R3/submission/team10_R3_defense.zip`を既存の非公開GitHubへ保存する許可を得ています。検証済みのCSV・ZIP・生成コード・評価レポートは変更せず、CodaBenchへは送信しません。R3/verification.jsonはGitHub保存前の生成検証記録として保持しています。

**2026-09-17、ユーザー提供のR3公式結果を受けて[R4](main_round/anonymization/team10/R/R4/README.md)を作成しました。** 固定R3のC.csvへ、年齢・性別・喫煙による条件付き検査値分布の補正を適用し、変位をCox係数の零空間へ射影して発症リスクを保持します。強度0.5、追加Cox補正なし。22候補を比較し、ローカルU=0.8683850877951735（U_gen=0.9319172595024955、U_spec=0.9136141267012007、U_rare=0.8683850877951735、U_valid=1）を採用。R3からUの改善は0.002260596と小幅ですが、全4指標は同等以上です。21テスト成功、提出ZIP作成。最終検証はR4/verification.jsonを参照。R1〜R3は変更せず、作成後の追加依頼によりR4一式と`main_round/anonymization/team10/R/R4/submission/team10_R4_defense.zip`を既存の非公開GitHubへ保存する許可を得ています。検証済み成果物のバイトは保持し、verification.jsonは保存前の生成検証記録として残しています。CodaBenchへは送信しません。R3の参照MIA=0をR4の攻撃耐性へ流用しません。

旧ルートの実行コード・データ・研究記録は、相対配置を保って `prelim/workspace/` に移動しました。
過去の `python -m attack_phase_...` コマンドは、先に `cd prelim/workspace` して実行します。
詳細引き継ぎに残る過去の提出推奨・残回数は履歴であり、現在の指示ではありません。
旧学習資料は初期v1の説明で、最終v9の仕様書ではありません。

本戦・加工フェーズの配布原本6ファイルは、2026-09-16のユーザーの明示的依頼に基づき、既存の非公開GitHubリポジトリへの保存対象としました。2026-09-17には、team10の `participant_data/token.txt` と `模擬提出データ/token.txt` も同じ非公開リポジトリへ保存するよう明示的に指定されました（いずれも `main_round/anonymization/team10/` 配下）。同日の追加許可により、`S/S1/submission/team10_S1_defense.zip` も同じ非公開リポジトリへの保存対象です。その他の実トークン・トークン入りZIP・予備戦の実配布B/B_self/A_bgは新たにアップロードしません。トークンの内容は文書やログに記載しません。
S2完了後の同日の依頼により、`S/S2/submission/team10_S2_defense.zip` も現在の成果物一式とともに同じ非公開リポジトリのmainへの反映対象です。それ以外のトークン入りZIPへの許可を追加するものではありません。
既存の非公開Git履歴を残すという所有者の判断は維持し、履歴書き換えやforce pushはしません。
