# リポジトリ構成と移動先

予備戦は `prelim/`、本戦は `main_round/` です。リポジトリ直下には共通の案内、Git 設定、ライセンスなどを残しています。

2026-09-30の最新配置：共通の模擬問題生成・採点を [`main_round/mock_test/`](main_round/mock_test/README.md) へ分離しました。複数抽出をスイートとして保存し、各作業者の攻撃を同じ問題へ適用します。

```text
main_round/
├── mock_test/                     # 共通の模擬テスト
│   ├── generate.ps1 / evaluate.ps1 # 問題の保存・プラグイン評価
│   ├── local_eval/                # 生成・読込・採点コード
│   └── local-output/              # 新しい問題・評価結果（Git対象外）
└── attack/
    ├── attack_TS/                 # TSの攻撃・モデル・過去の評価記録
    ├── Attack_M/                         # Mの攻撃・モデル・過去の評価記録
    └── targets/・official-kit/など # 共通配布物・検証・Python環境
```

`attack/attack_TS/local_eval/`・`generate.ps1`・`evaluate.ps1` は `mock_test/` へ移動しました。Pythonパッケージ名 `local_eval` と既存のTS/Mのモデル・予測・評価記録は維持します。新しい共通ツールの出力は `mock_test/local-output/` へ保存し、過去の記録に残るパスは当時の履歴として扱います。

以下は同日までの配置変更の履歴です。本人（TS）の本戦攻撃は [`main_round/attack/attack_TS/`](main_round/attack/attack_TS/README.md) にまとめました。当初は `attack_research/`・`local_eval/`・既存の`local-output/`・`evaluate.ps1`・`generate.ps1`を同フォルダーへ移動しました。その時点の対応・照合は `attack_TS/layout_migration.json` に残しています。現在の共通ツールの場所は上記を参照してください。

2026-09-30追加：作業者Mの本戦攻撃は [`main_round/attack/Attack_M/`](main_round/attack/Attack_M/README.md) にまとめます。Mの第1版は `Attack_M/v1/`、予測CSV・診断・検証記録は `Attack_M/local-output/v1/`、初期作戦は `Attack_M/strategy/` です。旧 `attack/v1/`・`attack/local-output/v1/` の保存対象をM配下へ移し、版名v1と保存CSVのバイトを保持しました。旧パスのローカル互換コピーはGit対象外です。

2026-09-27追加：[main_round/attack/](main_round/attack/README.md) に本戦の攻撃配布物・固定版キット・実行と検証の入口を配置しました。ユーザーの明示的依頼により、`downloads/` の配布原本、`targets/` の本戦攻撃対象、`official-kit/starter/` の固定版実行ソース、`reference_baseline/` の参照CSV・検証記録も既存の非公開リポジトリに保存します。専用Pythonの仮想環境・キャッシュ・今後の `local-output/` はローカル管理です。

以下のツリーは予備戦整理時点の履歴です。最新の本戦作業は上記リンクと[引き継ぎ](CODEX_HANDOFF.md)を参照してください。

2026-09-09追加：[予備戦の結果公開](prelim/results/20260909/README.md)に確定結果・公開原本・検証記録、[本戦ルール](main_round/rulebook/README.md)にmain-process-20260912のPDF・変更点を保存しています。以下の移動先一覧は予備戦整理時の記録です。

## 主要ファイル構造

```text
pwscup2026-kit-private/
├── README.md                         # 最初に読む案内
├── REPOSITORY_LAYOUT.md              # この構成・移動先一覧
├── AGENTS.md / CODEX_HANDOFF.md       # 現在の作業ルール・引き継ぎ先
├── LICENSE
├── .gitattributes / .gitignore
├── test_prelim_workspace.py          # ルートから予備戦テストを実行する入口
├── prelim/                          # 以下はすべて予備戦
│   ├── README.md
│   ├── anonymization/               # 匿名化フェーズの提出物
│   │   ├── README.md
│   │   └── anonymization-submitted-v5/
│   │       ├── .gitattributes
│   │       └── C.csv
│   ├── attack/                      # 攻撃フェーズの引き継ぎ用ファイル集
│   │   ├── README.md
│   │   ├── manifest.json            # 元候補との対応・ファイルハッシュ
│   │   ├── final/                   # 提出済み v9 の F_mia.csv / F_aia.csv
│   │   ├── candidates/              # その他 17 候補の CSV ペア
│   │   └── results/                 # 公式フィードバック 6 件と最終結果説明
│   ├── old_attack_study_materials/  # 初期手法の旧学習資料
│   │   ├── README.md
│   │   ├── PWSCup2026_今回の攻撃手法_図解ガイド.docx
│   │   └── PWSCup2026_C1実データ計算ウォークスルー.docx
│   └── workspace/                   # 予備戦の再現用作業環境
│       ├── README.md / AGENTS.md / CODEX_HANDOFF.md
│       ├── CODEX_START_PROMPT.md / LOCAL_SETUP_JA.md
│       ├── attack_phase/            # 初期の攻撃実装
│       ├── attack_phase_daybreak/
│       ├── attack_phase_daybreak_independent/
│       ├── attack_phase_v2/ ... attack_phase_v9/
│       │   └── （各版の実装・テスト・結果説明）
│       ├── opendp_aim/              # 匿名化の実装・設定・実験
│       ├── my_solution/             # 防御の作業用実装
│       ├── submissions/prelim/team24/
│       │   └── attack-candidate-*/  # 再現用の元候補 18 組
│       ├── team-shared/prelim/attack-targets/
│       │                            # 予備戦の共有攻撃対象
│       ├── local-output/            # 保存対象に選んだ実験・解析記録
│       ├── research_archive/        # 全作業アーカイブの説明・マニフェスト
│       ├── archive_prelim_work.py   # アーカイブ選別・検証
│       ├── starter/                 # 予備戦当時のキット・検証器・参照実装
│       ├── participant_data/        # 予備戦当時の練習用データ
│       ├── notebooks/               # 当時のキット付属教材
│       ├── rulebook/                # 当時のルールブック
│       └── setup-local.ps1 / run-practice.ps1
└── main_round/                      # 本戦用。加工フェーズの実データ配置済み
    ├── README.md
    ├── anonymization/
    │   └── team10/                  # 本戦コホート10（予備戦は24）
    │       ├── README.md / data_manifest.json
    │       ├── S/ / R/ / M/ / L/    # 作業者別モデル格納先
    │       ├── 模擬提出データ/       # 予備戦v5のC.csvと本戦team10のtoken.txt
    │       └── participant_data/    # 配布原本6ファイルと明示許可されたtoken.txt
    └── rulebook/                    # main-process-20260912 PDFと更新確認
```

細かなコード・実験ファイルは省略しています。`.venv/`、キャッシュ、上記team10の2ファイル以外の実トークン、トークン入り ZIP など、ローカル専用の無視対象もこのツリーには載せていません。

## 旧パスからの移動先

すべてリポジトリ直下からの相対パスです。

| 整理前 | 整理後 |
|---|---|
| `submissions/prelim/team24/submitted-v5/` | `prelim/anonymization/anonymization-submitted-v5/` |
| `study_materials/` | `prelim/old_attack_study_materials/` |
| `attack_phase/`、`attack_phase_v2/`〜`attack_phase_v9/`、Daybreak 系 | `prelim/workspace/` 以下の同名ディレクトリ |
| `submissions/prelim/team24/attack-candidate-*/` | `prelim/workspace/submissions/prelim/team24/attack-candidate-*/` |
| `opendp_aim/`、`my_solution/` | `prelim/workspace/` 以下の同名ディレクトリ |
| `team-shared/`、`local-output/`、`research_archive/` | `prelim/workspace/` 以下の同名ディレクトリ |
| `starter/`、`participant_data/`、`notebooks/`、`rulebook/` | `prelim/workspace/` 以下の同名ディレクトリ |
| `setup-local.ps1`、`run-practice.ps1`、`LOCAL_SETUP_JA.md`、`CODEX_START_PROMPT.md`、`archive_prelim_work.py` | `prelim/workspace/` 以下の同名ファイル |
| 従来の `README.md`、`AGENTS.md`、`CODEX_HANDOFF.md` | 予備戦向けの内容を `prelim/workspace/` に保存。直下には新しい案内を配置 |
| `prelim/attack/` | 場所は変更なし。移動後の元候補への参照を更新 |
| `.gitattributes`、`.gitignore`、`LICENSE` | リポジトリ直下に維持 |

## 再現するときの作業場所

テストだけならリポジトリ直下の `python -m unittest discover -v` でも、`test_prelim_workspace.py` を通じて予備戦の全テストを実行できます。

予備戦のコード、元候補、キット、実験記録は相互参照が多いため、`prelim/workspace/` にまとまった作業環境として移動しました。内部の相対配置を維持し、今回の整理では攻撃アルゴリズムや提出 CSV を変更しません。

従来の手順にある「リポジトリ直下」は、予備戦の実行については **`prelim/workspace/`** と読み替えます。予備戦用の依存関係を導入した Python で、例えば次のように実行します。

```powershell
# リポジトリ直下から
cd prelim/workspace
python -m unittest discover -v
```

環境構築の詳細は [`prelim/workspace/LOCAL_SETUP_JA.md`](prelim/workspace/LOCAL_SETUP_JA.md)、当時のキットの使用方法は [`prelim/workspace/README.md`](prelim/workspace/README.md) にあります。新しいチェックアウトで仮想環境がなければ、先に構築してください。過去の記録に個人環境の絶対パスがある場合は、そのまま流用せず実際の配置に合わせます。

`attack/final/` と `attack/candidates/` は閲覧・引き継ぎ用です。再現用の元候補は `workspace/submissions/prelim/team24/` にあり、両者は別々の実験結果ではありません。対応は [`prelim/attack/manifest.json`](prelim/attack/manifest.json) を参照してください。

## 保存範囲と現在の状態

- 予備戦の最終提出は **v9 guarded、公式結果（ユーザー共有）0.3407**。提出枠は残っていません。v7 単独の提出結果や本戦での性能はありません。
- `participant_data/` と `starter/` は、予備戦当時の練習キットの保存物です。本戦のキット・実データとして分類したものではありません。
- 実配布の team24 の B/B_self/A_bg は、ユーザーの Downloads 配下などのリポジトリ外の保存先に残します。練習用の同名ファイルと混同せず、新たな Git 管理対象に加えません。
- 本戦・加工フェーズの team10 配布原本6ファイルは、2026-09-16のユーザーの明示的依頼に基づき `main_round/anonymization/team10/participant_data/` に配置し、既存の非公開GitHubへの保存対象としました。2026-09-17の明示的指定により、同じteam10配下の `participant_data/token.txt` と `模擬提出データ/token.txt` も非公開リポジトリで管理します。
- 上記2ファイル以外の実トークン、トークン入り ZIP、仮想環境・キャッシュは新規のアップロード対象外です。トークンの内容は文書やログに記載しません。既存ローカルファイルは保持し、過去の Git 履歴は書き換えていません。
- `local-output/` は原則ローカル出力先ですが、選別済みの研究用テキスト・CSV・JSON・スクリプトはアーカイブされています。保存範囲は [`prelim/workspace/research_archive/README.md`](prelim/workspace/research_archive/README.md) とマニフェストが正本です。
- 旧学習資料は初期手法を理解するために残します。最終 v9 の確認は [`prelim/attack/results/FINAL_RESULT.md`](prelim/attack/results/FINAL_RESULT.md) と [`prelim/workspace/attack_phase_v9/README.md`](prelim/workspace/attack_phase_v9/README.md) を優先してください。

## 整理時の確認

全177テストと、v9生成器・ZIP作成器のコマンド起動を確認しました。攻撃ファイル集の43件は、移動後の元ファイルとサイズ・SHA-256が一致します。提出CSV、公式結果JSON、Word原本、配布キットのバイト列は保持しています。公開前には秘密情報検査とGitステージのmanifest照合も行います。今回、公式提出や攻撃アルゴリズムの変更は行っていません。
