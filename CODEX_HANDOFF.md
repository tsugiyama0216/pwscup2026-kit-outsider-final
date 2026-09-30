# PWSCUP2026 引き継ぎ

## 共通模擬テストの現在の入口

GitHub保存前の統合確認では、共通テスト29件・M v1の15件・M v2の24件が成功しました。M v2の固定依存31件とv1の保持対象65件を照合し、保存済み3抽出×7方式の採点42件も統合前と一致しています。[今回の統合検証記録](main_round/mock_test/github_integration_20260930.json)を参照してください。過去の移行・評価JSONは作成当時の記録として保持しています。

続く2026-09-30の依頼により、この共通ツールと説明・検証記録を既存の非公開GitHubのmainへ保存する許可を受けました。生成済み模擬問題と通常の評価出力はローカル管理を継続し、GitHub側で追加されたMのv2と固定依存の互換性を維持して統合します。

2026-09-30、ユーザーの依頼で模擬問題生成・採点を攻撃実装から分離し、[main_round/mock_test/](main_round/mock_test/README.md) に配置しました。`generate.ps1` は既定3抽出（`-Count` で変更）を保存し、`evaluate.ps1 -CaseDirectory` は保存した単一問題または複数抽出スイートへM・TS等のプラグインを適用します。複数の保存先も指定できます。新しい共通出力は `mock_test/local-output/`、各攻撃の実装・モデル・過去の結果は `attack/M/`・`attack/attack_TS/` に保持します。パッケージ名は `local_eval` のままです。

```powershell
Set-Location main_round/mock_test
..\attack\setup.ps1 # 専用環境がないPCのみ
.\generate.ps1 -Split dev -Seed 20260927 -Count 3
.\evaluate.ps1 -CaseDirectory .\local-output\shadow-cases\dev-20260927-n3 -Plugin ..\attack\M\v1\model.py
```

同じ保存スイートを使って平均・最小最大・問題別・方式別・公式参照との勝敗を比較します。候補選択はdevで行い、候補を固定してからholdoutを確認します。人物分割のholdoutは既使用であり、新seedでも独立した未知人物とは呼べません。単一抽出の主5条件、既定3抽出の主15条件、v88の固定25条件は抽出条件を確認して区別します。この整理ではモデル再学習・過去の結果の再生成・公式提出を行いません。このPCの本戦専用環境は構築済みで、原本94件・公式指定29ライブラリの確認に成功しました。以下のv88依存不足は未解決です。

**2026-09-30、ユーザーが「アップロード＝mainへ統合して保存」と明示。** 今後このリポジトリのアップロード依頼は、指定成果物をmainへ統合・pushするところまでを含む。Mのv2も対象。先行保存`a5706e5`の固定コード・設定・CSV・生成時記録を保ち、TSと共通模擬テストの配置と両立させる。`attack_workspace.py`・`attack.ps1`・Mのv1評価コード・`attack/local_eval/`の3ファイルはv2固定時のバイトを保持する。`attack/local_eval/`はM v1/v2専用の互換スナップショットで、共通評価器の正本は`mock_test/local_eval/`。共通ツールが必要なパスと保存先属性を初期化し、TS側もその処理を利用する。ハッシュ検証の迂回や再選択はしない。M v2を最初にmainへ統合した時点の記録は[統合検証記録](main_round/attack/M/main_integration_20260930.json)を参照。TSの不足依存に関する次の引き継ぎは引き続き有効。

## V88_CROSS_PC_HANDOFF

**別PCで取得した担当者・エージェントへ：TSのv88は依存コードが不足しており、まだ実行一式がそろっていません。** 2026-09-30に取得した `main`（`fad8253860066b43bbed38e33f3fb5bbacf56a12`）で確認しました。`attack_TS/` への整理前から存在した不足です。

- **不足しているコード：** `prelim_transfer.methods`。必要な関数は `load_legacy()` と `build_aia()` です。元の作業PCの `main_round/attack/prelim_transfer/` または `main_round/attack/attack_TS/prelim_transfer/` を探索先の候補とし、元コードとそこから読み込む依存コードを一緒に確認してください。元PCでの存在や回収はまだ確認していません。
- **必要な理由：** [v88](main_round/attack/attack_TS/attack_research/v88/predict.py) は過去版を順に呼び、その先の [v10](main_round/attack/attack_TS/attack_research/v10/predict.py) が初期化時に `load_legacy()`、予測時に `build_aia()` と `modules['structured'].stratified_half_selection` を使います。現在は `CombinedAttack()` の初期化が `ModuleNotFoundError: No module named 'prelim_transfer'` で停止します。
- **照合成功の意味：** `attack_research.published_v88.check` は、マニフェストに列挙されたファイルと原本のハッシュを確認します。列挙されていない依存モジュールの有無や予測の完走は確認しません。106ファイル・原本94件の照合成功、学習済みモデル11件の読み込み成功を、v88全体の実行成功と扱わないでください。
- **配置：** TSの正本は `main_round/attack/attack_TS/`、Mの攻撃は `main_round/attack/M/`。回収する `prelim_transfer/` はTS配下で既存の `from prelim_transfer.methods import ...` が解決するように配置し、元コードが参照する予備戦モジュールのパスも確認します。実装を推測して置き換えず、元のv88を再現できるコードを回収してください。
- **実行環境：** 本戦専用 `.venv/` はGitに含まれません。別PCでは `main_round/attack/setup.ps1` で固定版公式環境を構築・確認します。このPCでは2026-09-30に専用環境を構築し、原本94件と公式指定29ライブラリの確認に成功しました。当初の予備戦Pythonには公式指定と異なる依存版が6件あり、v88全体の実行確認には使用していません。専用環境の構築後も、不足コードの復旧とv88全体の実行確認は未完了です。

復旧を依頼された担当者は、元コードと再帰的な依存先を回収し、相対パスを確認してから専用環境を構築します。その後、ファイル照合、TS配下からのimportと `load_legacy()`、v88の初期化、最後に固定条件の `benchmark` の順で確認してください。既存の学習済みモデル・選択記録・予測CSVは保持します。回収した実行時コードを現在の `manifest.json` の検証対象に追加し、`manifest_before_attack_TS.json` と過去の検証JSONは元の記録として残します。復旧後の結果は日時・使用コミット・コマンドとともに新しく記録し、この「不足中」の案内も更新してください。

復旧後の最小確認例（リポジトリ直下から。**現状では依存コード不足で途中停止します**）：

```powershell
Set-Location main_round/attack/attack_TS
..\setup.ps1
..\.venv\Scripts\python.exe -m attack_research.published_v88.check
..\.venv\Scripts\python.exe -c "import attack_research; from prelim_transfer.methods import load_legacy, build_aia; modules = load_legacy(); assert callable(modules['structured'].stratified_half_selection)"
..\.venv\Scripts\python.exe -c "from attack_research.v88.predict import CombinedAttack; CombinedAttack(); print('v88 initialization OK')"
..\.venv\Scripts\python.exe -m attack_research.published_v88.benchmark
```

元の記録にあるbenchmarkの参照値は、固定25条件でMIA平均 `0.7307692307692308`、AIA平均 `0.8473837794327597`、両方1.0が `8/25` です。別PCで復旧後に再現を確認する値であり、このPCで今回実行した結果や公式得点ではありません。

この復旧案内を作成した時点の依頼では、この案内と前段の `attack_TS/` 整理を同じ非公開GitHubで共有し、別PCのclone/pullで読めるようにしました。不足コードの回収・復旧と公式提出はまだ行っていません。現在の共通模擬テストの配置は冒頭を参照してください。

## これまでの作業記録

**2026-09-30、本人の指定によりTSの攻撃を [attack_TS](main_round/attack/attack_TS/README.md) に分離。** `attack_research/`、`local_eval/`、保存済み`local-output/`、`evaluate.ps1`・`generate.ps1`を `main_round/attack/attack_TS/` へ移動。Mの独立攻撃は従来の `M/` に保持し、Mが再利用する模擬評価コードの参照先だけを更新する。配布原本・公式キット・検証・Python環境は共通の親フォルダーに置く。学習済みモデル、予測CSV、過去の評価JSONは保持し、対応と照合は `attack_TS/layout_migration.json` に記録する。今回の依頼はローカル整理であり、新たなGitHub送信・公式提出・モデル再学習は行わない。

**2026-09-30、完成済みMのv2をユーザー依頼で非公開GitHubへ保存。** 保存ブランチは`codex/m-v2`、基点は`fad8253`。取得済み最新mainの`45511cf`は共有評価コード・検証補助・v1評価コードの配置を変更しており、v2が固定した依存ハッシュと一致しない。先行保存はコード・設定・CSV・生成時記録を保持するため専用ブランチで行った。その後、ユーザーの明示に従い、上記の互換対応を含めてmainへ統合する。詳細は[Mのv2の案内](main_round/attack/M/v2/README.md)と[保存時検証](main_round/attack/M/local-output/v2/github_upload_verification.json)。生成時JSONの未送信フラグは当時の記録として保持する。

**2026-09-30、ユーザー依頼で作業者Mの本戦攻撃[v2](main_round/attack/M/v2/README.md)を作成・検証。** コードと固定設定は`main_round/attack/M/v2/`、全結果は`M/local-output/v2/`。v88・attack_researchは参照せず、v1の65ファイルをバイト保持。MIAはv1密度比から候補ごとの22相手中央値を強度1で減算、AIAは県・性別候補40未満かつ県固定距離が性別のみ距離の1.5倍超なら性別のみへ切替（guarded40）。厳密な非共有163人のMIA=0を保持。調整・非候補独立合成ストレス検査後に全依存と方式を固定し、別抽出の最終確認を実施、確認後再調整なし。MIA世界等重み平均.778927→.815411（18世界全改善、主351対象条件173改善169同等9悪化）。AIAφ未控除.274991→.370155（主135条件25改善108同等2悪化、最悪−.103408）。両者は別実験で合算しない。人物分割は過去研究で既使用、MIAの非候補共有・AIAの県集中も模擬条件で公式得点予測ではない。22相手MIA1049行・AIA7172行、公式形式とCSV再読込・2回生成バイト一致、24テスト、原本94件不変を確認。AIA変更はC6=297、C8=74、C14=92、他19対象はv1と同一。入口は`attack.ps1 M-v2-*`。選択済みコード・JSONはハッシュ固定されているので改修は次版で行う。公式提出・試行消費・提出ZIP作成・GitHub送信は行っていない。

**2026-09-30、ユーザー指定により「作業者Mのv1」と分かる[配置](main_round/attack/M/README.md)へ整理。** コードは`main_round/attack/M/v1/`、保存予測・検証は`M/local-output/v1/`、初期作戦は`M/strategy/`。版名はv1を保持。共有データ・公式キット・他作業者の成果は移動しない。実行入口は`attack.ps1 M-v1-*`（旧`v1-*`も同じMのコードへ接続）。旧CSV・モデル・生成時JSONはバイト保持し、[layout_migration.json](main_round/attack/M/layout_migration.json)へ照合を記録する。進行中の研究が読む旧パスはローカル互換コピーとして保持し、GitHubの正本はM配下とする。

**2026-09-30、ユーザーが完成済み本戦攻撃v1の非公開GitHub保存を依頼。** v1コード・実行入口・説明、`M/local-output/v1/`の診断・調整・最終確認・予測CSVと検証記録、初期作戦と公開手掛かり監査を保存対象とする。固定CSV・モデル・生成時記録のバイトは保持し、後続研究と無関係なローカル変更は含めない。保存時の検証は[M/v1/github_snapshot_verification.json](main_round/attack/M/v1/github_snapshot_verification.json)。生成時記録の`github_push_performed:false`は当時の履歴として維持。CodaBenchへの提出やトークン入りZIPの作成・保存は行わない。

**2026-09-30、自分の本戦攻撃[v1](main_round/attack/M/v1/README.md)を実装・ローカル検証。** ユーザー依頼の順に22対象の加工診断、独立v1、具体的検証を完了。v88・attack_researchは参照せず、一般的な模擬データ生成・公式採点のみ再利用。採用は追加属性の近傍距離比MIA＋公式AIA、自己challengeの厳密一致から導く非共有163人を本番MIAで0。調整後固定した15主条件の最終確認は公式1.089288→v1 1.255955（MIA＋AIAのφ未控除代理、5改善8同等2悪化）。人物分割は別研究で既使用で、本戦得点予測ではない。原本94件、15テスト、MIA1049×22・AIA7172行の公式形式、全件2回生成のバイト一致を確認。CSVと記録は `main_round/attack/M/local-output/v1/`、コード・選択は`main_round/attack/M/v1/`。入口は`attack.ps1 M-v1-*`。公式提出・ZIP作成・GitHub送信は行っていない。

**2026-09-27、[本戦・攻撃環境](main_round/attack/README.md)を整備し、ユーザーが現在の状態の非公開GitHub保存を依頼。** `main-attack-20260928`の配布原本・固定版公式キット・team10用実行と検証の入口を保存対象とします。マニフェスト94件のハッシュ照合、公式指定29ライブラリ、本戦22対象の参照攻撃（MIA 1,049行、AIA 7,172行）の検証は完了しています。保存した参照出力と検証記録は `main_round/attack/reference_baseline/`、実行入口は `main_round/attack/attack.ps1` です。期間は9月28日09:00〜10月11日09:00 JST、採点に通る提出は最大8回。参照攻撃は動作確認用で、最適化済み候補や公式攻撃力ではありません。仮想環境・キャッシュ・新たな実トークンのコピー・提出ZIPは今回の追加対象に含めず、既存の明示許可済み履歴を維持します。CodaBenchへの提出は行っていません。

**自己Cの照合を完了しました。** 配布 `main_round/attack/targets/C_10.csv` は、事務局メールのSHA-256と、既存の最終選択モデル [`M/M1_v5/C.csv`](main_round/anonymization/team10/M/M1_v5/C.csv) の両方に一致します（SHA-256 `7bcc96898b3d6bf4af8b7966d481fd0136dc64bbf3b68d0a5dd751a0b859c7db`）。事務局掲載の提出IDは943347です。環境整備時に残っていた「最終提出原本との照合未確認」は解消しました。自己10は攻撃対象から除外します。以下の記録は各記載日時点の履歴です。

**2026-09-25、ユーザーが現在の成果物のGitHub保存を依頼。** 現在のteam10作業一式と、検証済みV5提出ZIPを既存非公開リポジトリのmainへ保存する許可。V5 ZIPの従来のローカル限定条件はこの追加依頼で更新。M1_v3〜v5、DOMIAS/AIA再評価、DPCopula ε15、Rモデルの監査記録を対象とし、直近の選定根拠は[M/evaluation/model_selection_20260925](main_round/anonymization/team10/M/evaluation/model_selection_20260925/README.md)へ集約。旧V3 ZIP、AIA private/、キャッシュなど既存の除外は維持し、CodaBenchには送信しない。作成ログの`github_pushed:false`はZIP作成時点の記録として改変しない。

**2026-09-25、ユーザーが提出モデルをM1_v5へ変更し、[提出用ZIPと作成ログ](main_round/anonymization/team10/M/M1_v5/FINAL_SUBMISSION.md)を作成。** 最新の指定が以前のM1_v3選択に優先する。`M/M1_v5/submission/team10_M1_v5_defense.zip`直下は評価済みC.csvとteam10原本token.txtのみ。1,049行×13列、固定公式キットのZIP/CSV形式検証・配布B行数照合・CRC・元CSVとのバイト一致に合格。CSV SHA256=7bcc96898b3d6bf4af8b7966d481fd0136dc64bbf3b68d0a5dd751a0b859c7db、ZIP SHA256=3a5b9bae363b8f9ba8f86d17d8a03b76000e26c69ee1b5eaa26181107cbf7183（64,458 bytes）。tokenは検証済みV3 ZIPからメモリ内で取得し、配布原本のGitオブジェクトIDに一致。値は出力しない。V3/V4/V5のCSV・V3 ZIP・V5モデル不変。CSVの再生成・再学習・再採点は不要で実施せず。作成ログはV5/submission/CREATION_LOG.mdおよびcreation_log.json。今回の許可はローカルZIP作成のみ。CodaBench送付・GitHubアップロードはしていない。

**2026-09-25、引き継ぎ資料を受けて[M1_v3・V4・V5を再評価](main_round/anonymization/team10/M/evaluation/aia_reassessment_20260925/README.md)。** 公式AIAの公開7属性だけで予測し、標的163・対照163、各側希少13・発症16・希少発症4、県/性/5歳帯/発症/希少を対応抽出。公式最近傍とQI限定多視点の最大リスクを採用、希少SE z=1.64。本戦Pi・challenge・非公開対照は未入手で、Bから標的・公開Aから対照、対応可能なB999人のみ（50人除外）。MIAとは別々の代理集団で、真の共有/非共有ブロックを同時再現したものではない。φ10=0.026887はユーザー提供済みだが、この疑似challengeに適用確認できず主結果は無控除、別列に仮定シナリオ。有用性は再採点してV3=.8426441632、V4/V5=.8745376811。全DOMIAS確信度を再計算し、保存20,000置換をハッシュ・観測値・95%点で検証して再利用。主seed20260924はV3/V4 Anon=.5979020979、V5=.8461538462（旧.9303084031から修正）、AIAは3候補とも.1538461538。仮に提供φを控除したV5はAnon/総合=.8730408462。9seedでV5対V4総合は5改善4同点、最低AnonはV5=.6328671329・V3/V4=.5629370629。QI限定全B/全A診断ではV5 AIA=.043092679がV4=.029854781より高く、AIAの一貫した改善は主張しない。5テスト成功、全入力ハッシュ・提出ZIP不変。正式選択V3を維持、モデル生成・GitHubアップロード・CodaBench送付なし。新評価のCSV/固定条件/検証記録を同フォルダーへ保存、個人単位の抽出位置はprivate/でGit対象外。

**2026-09-24、M1_v4の改善候補[M1_v5](main_round/anonymization/team10/M/M1_v5/README.md)を作成。** V4の公開検出希少65行のBMI・TG・ALT（195セル）を公開A_bgの標準化7列の近傍50件へ最大35%補間。希少境界を越える11行では強さを抑制。通常984行と他10列・転帰は文字列表現まで保持。3/6列×4強さの8候補を、既知6seed・U≥.86・AIA≤.10・V4固定chance95の最悪総合代理値で選び、three_35採用。U=.8745376811でV4と同一、gen=.9200261447、spec=.9280276668、rare=.9669160992、valid=.8745376811。AIA代理.0743580553→.0696915969。採用CSVを凍結後、既知6＋新規3seed（20261011/12/13）を候補別20,000置換で比較。主seed Anon .597902→.930308、総合代理9条件中5改善4同点。新規Anon範囲はV4=.562937～.925642、V5=.632867～.930308。新規条件は同じB/Aの候補再抽出で独立データではない。追加多視点MIAの生TPRは一部で0→.001923へ微増、全攻撃での改善を主張しない。公開Aの生成・背景・対照への共用バイアスあり、非DP・公式未確認。4テスト、公開計算と保存モデルの双方でバイト再現、公式CSV検証成功。V5 SHA256=7bcc96898b3d6bf4af8b7966d481fd0136dc64bbf3b68d0a5dd751a0b859c7db。既存V3 CSV/ZIPとV4不変、最終選択V3を維持。V5 ZIP作成・GitHubアップロード・CodaBench送信はしていない。

**2026-09-24、ユーザーの改善依頼で[M1_v4](main_round/anonymization/team10/M/M1_v4/README.md)を別候補として作成。** M1_v3の静的10列とdeathを各行で保持し、希少検出区分内のtime/onsetを4交換、8行16セル変更。R8由来の集計Cox補正、4設定からU改善かつAIA非増加のguard225を選択。U=.8426441632→.8745376811、U_spec=.9348792246、gen/rare不変、valid=.8745376811へ微減。AIA代理=.0750979769→.0743580553。全6,192候補の参照DOMIAS確信度はv3と完全一致。既知3seed＋事前指定の新規3seed（20261001/2/3）、各20攻撃者と20,000置換で総合代理スコア3改善・3同点、最悪.597902は不変。追加の転帰を使う多視点MIAは6seedで生TPR0、AUCはほぼ同じ。新規seedは同じデータの候補再抽出で独立データではない。私的集計適合・選択で非DP、公式結果未確認。CSVはd32640fd8a85866121235053dc7c02c29ab87979c8fd5e0013aae9ca49db7201、形式検証・バイト再現・3テスト成功。generate.pyは固定v3とmodel.jsonのみで再現。最終提出選択M1_v3、同CSV・ZIPを保持し、v4の提出ZIP作成・GitHubアップロード・CodaBench送信は実施していない。

**2026-09-24、ユーザーがM1_v3を最終提出モデルとして指定し、提出用ZIPと作成ログを作成。** [提出案内](main_round/anonymization/team10/M/M1_v3/FINAL_SUBMISSION.md)から参照。`M/M1_v3/submission/team10_M1_v3_defense.zip`の直下は受領原本と同一のC.csvとteam10のtoken.txtのみ。1,049行×13列、公式キット固定コミット3dd71bd3436f2abfdbb64da97e860f22729ea054によるZIP・CSV形式検証と配布B行数照合に合格。C SHA-256=d425f6d612f0d3f0d8af3ee0a7f042d846c25ad9555966b437f3623773c175a1、ZIP SHA-256=5be38cda5fb0a234d8500ce7bac1148ae62cbc65f0607e1c5124d2ccfcf71068。受領7原本は不変、再生成・再学習なし。既存ローカルU=.842644163219807、今回再採点なし。作成日時・来歴・検証結果はsubmission/CREATION_LOG.mdとcreation_log.json。token原本が部分チェックアウトで未展開のため既存R12 ZIPからメモリ内取得し、Git記録の配布原本オブジェクトIDと一致確認。token値は出力しない。今回の許可はローカル提出用ファイル作成であり、GitHubへのアップロード・CodaBench送信は実施していない。本戦形式のDOMIAS代理比較はM/evaluation/domias_main_round_20260924/README.mdを参照。過去の候補方針より今回の最終選択を優先する。

**2026-09-19、R11の有用性を維持する簡略生成モデル[R12](main_round/anonymization/team10/R/R12/README.md)を作成。** R11公式フィードバックをR12/R11_user_reported_result.jsonに保存。R11の合成順位と固定7列を学習し、通常993行の6検査値を各256分位点で線形補間。希少56行は保持し、境界を越える通常8行はR11値を保持する。計12候補・2バッチから最少分位点の適格候補を選択。985行・5,898セルが変化。ローカルU=.9962017963269137でR11と同一、gen=.9974261201143946、spec=.9975884876119974、rare=.9962017963269137、valid=1。U_spec単独は低下。生成は保存model.jsonだけで可能でCox・最適化は不要だが、R11をゼロから生成する複雑さは来歴として残る。4テスト成功、公式ZIP検証とモデルからのCSVバイト再現成功。提出ZIP `R/R12/submission/team10_R12_defense.zip` を含めこのPCと既存非公開GitHubへの保存をユーザーが明示。最終総合得点・R12の公式参照MIA/AUCは未確認。CodaBenchへは送信しない。R1〜R11と他モデルを保持。

**2026-09-19、R9の得点向上を目的に[R11](main_round/anonymization/team10/R/R11/README.md)を作成。** ユーザーはR9を参考に改善し、R11の提出データをteam10/Rに保存してこのPCとGitHubの場所を示すよう依頼。R10のAUC上昇目的は引き継がず、凍結R9から生成。希少群6検査値の経験分位点位置0.75・強度1.0へ調整し、7行の検出区分を復元。希少56行を固定して、多数派993行の6検査値を集計Cox・分布・相関へ250反復で補正。16候補・3バッチから全ファセット非低下かつ最大Uを選択。U=.9962017963269137（gen=.9973307912297426、spec=.9995605261019739、rare=.9962017963269137、valid=1）でR9のU=.9951442572649957を改善。5テスト成功。各行の年齢・性別・地域・喫煙・時刻・発症・死亡と希少区分を保持。最適化は反復上限終了。公式AUC・参照MIA・最終総合得点は未確認。提出ZIP `R/R11/submission/team10_R11_defense.zip` を含むこのPCと既存非公開GitHubへの保存をユーザーが明示。最終形式・再現確認はR11/verification.json。CodaBenchへ送信せず、R1〜R10と他モデルを保持。

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
