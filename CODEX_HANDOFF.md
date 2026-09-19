# PWSCUP2026 引き継ぎ

**2026-09-19、ユーザーの再保存依頼により[R10](main_round/anonymization/team10/R/R10/README.md)を復元し提出ZIPを追加。** 直前にR10を削除したが、最新依頼は「R10の提出データをteam10/Rに格納し、このPCとGitHubの保存場所を示す」。削除前コミット19148a6のR10全11ファイルとバイト一致を確認して復元し、CSVはSHA256=8e5797cbff0be3d312b7bd55584caef4ed91793fb7e5109e334849d6e4a658bbを保持。`R/R10/submission/team10_R10_defense.zip`を作成し、ZIP直下はC.csvと原本token.txtのみ。公式ZIP検証とZIPからの再採点はpackage_verification.jsonを参照。U=.9951442572649957、代理AUC=.7650158748635545、公式AUCは未評価。R9公式のAUC=.518515625と代理AUCは比較しない。以前ユーザーは攻撃側AUCの数値上昇を明示しており、匿名性改善を主張しない。今回の明示的依頼は提出ZIPを含む既存非公開GitHubへの保存を許可する。CodaBenchへの送信は行わない。既存の他モデルを保持。

**2026-09-18、M共通耐性評価の説明資料を拡充。** `evaluation/METRICS_GUIDE.md`に比較表の全項目、代理集合、運営DOMIAS KDE、多視点MIA、運営・多視点AIA、20,000回偶然補正、floor・balance・耐性余裕・希少MIAギャップ、現在の横断結論、判断手順、限界を記載。全12モデルの`results/<model>/DETAILS.md`に生成特徴、指標の意味、MIA/AIA内訳、完全一致、モデル固有の原因仮説、次のアブレーション、言えないことを追加。原因は断定せず、モデル差分と攻撃仕様からの仮説として明示。比較表とSUMMARYから詳細解説へリンクする。

**2026-09-18、R8公式結果を受けて[R9](main_round/anonymization/team10/R/R9/README.md)を作成。** R8公式はgen=.974715229174222、spec=.9724474721829109、rare=.9792811713362229、valid=1、MIA=0、AUC=.5330859375。R9は固定R8に地域件数の補正58行、検出区分別の年齢分位点補正332行を行い、分布・順位相関・Coxスコアと標準誤差を集計目標に6,294変数を300反復で同時補正。12候補・3段階で比較し、ローカルU=.9951442572649957（gen=.9970448045757865、spec=.999408266781119、rare=.9951442572649957、valid=1）を採用。全指標がR8以上で0.99超え。検出希少56、希少発症18、各行の時刻・発症・死亡を保持。10テスト合格、最終形式・再現検証はR9/verification.json。最適化は反復上限終了で完全収束を主張しない。独立ホールドアウトではなく、R9公式攻撃耐性は未評価。R1〜R8とM/S系列を保持。GitHubのM1追加コミット294f264をfast-forwardで取り込んだ。ユーザーはこのPCと既存非公開GitHubへの保存を指定し、`main_round/anonymization/team10/R/R9/submission/team10_R9_defense.zip`を含む。CodaBenchへ送信しない。

**2026-09-18、共通耐性評価をS1～S2・R1～R9・M1へ拡張。** GitHub mainからR9をfast-forward取得し、S/Rの成果物は変更せず、結果を`main_round/anonymization/team10/M/evaluation/`に保存。診断用screening floorはM1=.752153、R5=.581549、S2=.548413、R8=.440704、R1/R2=.426620、R3=.412535、R4/R7=.398451、R6/R9=.356197、S1=0。R9はU=.995144だがMIA β=.643803で耐性がボトルネック。S1以外はすべて`high-local-signal`で、全モデルともAIAより運営DOMIAS MIAが支配的。特に希少側TPRが主要課題。比較表へモデル別メモ、Uと耐性の調和平均balance、耐性余裕R-Uを追加し、`MODEL_DIAGNOSTICS.md`へ生成特徴・希少MIAギャップ・完全一致・結論を記録。全モデルでB完全一致とC内重複は0。これはB全行対A_bgの代理診断で、公式protection/Anon予測ではない。

**2026-09-18、M系列の共通耐性評価基盤を[追加](main_round/anonymization/team10/M/evaluation/README.md)。** 本戦ルール§6.2–6.3と固定キットに従い、運営DOMIAS MIA、運営QI最近傍AIA、予備戦v4/v9由来の多視点MIA/AIA、MIA 20,000回偶然補正、希少worst-caseを実装。公式challengeがないためB全行対A_bgの代理評価であり、team10のAIA φも未適用。モデル順位には候補構成差で飽和しやすいK=12正規化前のβとAIA Rから `診断用耐性=1-max(β,R)`、`screening floor=min(U,耐性)` を使う。S1はU=0、耐性=.968676。M1はU=.752153、MIA β=.164930、AIA R=.076456、耐性=.835070、screening floor=.752153だが、運営DOMIAS希少側の強いMIA信号により`high-local-signal`判定。M1を提出確定せず、後継でMIA低減を優先。これは公式protection/Anon予測ではない。

**2026-09-18、作業者Mの[M1](main_round/anonymization/team10/M/M1/README.md)を作成。** S1の固定DP合成Cへ、公開A_bgの順位依存構造を使うランク・コピュラ補正を適用し、公開希少ドナーと公開Coxリスク最近傍の転帰割当を再適用した。採用設定は強度1.0、seed=20260919、死亡重み0.31。ローカルU=0.7521530456176521（gen=.9214155715372894、spec=.7521530456176521、rare=.8183050847457627、valid=.8727753152771738）。1,049行・13列、希少66、欠損・重複なし、同一設定のCSVバイト再現と検証成功。生成処理はB/B_selfを読まないが、候補選択にはローカル自己採点を使用しているため、選択全体を事前固定DP機構とは主張しない。ユーザーはM1の既存非公開GitHubへの保存を依頼。M1のトークン入りZIP作成・CodaBench提出は許可されていない。

**2026-09-18、R7のユーザー提供公式結果を受けて[R8](main_round/anonymization/team10/R/R8/README.md)を作成。** R7公式はgen=.965034580679937、spec=.9568840160028894、rare=.9639190249630962、valid=1、MIA=0、AUC=.5175。R8は多数派コピュラ0.5、区分を保護する非線形分散補正、経験分位点へ戻す希少群コピュラ1.0、B検出希少群の集計発症率とCox係数への事象交換4回。63設定・9段階で比較し、全4指標がR7以上の最大U=.9724474721829108（gen=.974715229174222、spec=.9724474721829108、rare=.9792811713362229、valid=1）を採用。検出希少56、発症18件。17テスト合格、公式形式・再現検証の記録はR8/verification.json。R8はローカル値、独立ホールドアウトではなく、公式攻撃耐性は未評価。R1〜R7は保持。ユーザーはこのPCと既存非公開GitHubへの保存を指定し、`main_round/anonymization/team10/R/R8/submission/team10_R8_defense.zip`を含む。CodaBenchへ送信しない。

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
