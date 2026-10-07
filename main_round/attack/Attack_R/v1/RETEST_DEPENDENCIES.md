# retestに必要な凍結R v1ソース

このGitHub保存では、[retest](../../retest/README.md) の既定R v1評価に必要な固定依存を収録します。R v1はM v2 MIAとT v88 AIAの組合せで、モデル・選択・予測関数を変更していません。

| ファイル | retestで必要な理由 |
|---|---|
| `common.py` | リポジトリ基準パスと172件の固定依存ハッシュ照合 |
| `model.py` | 凍結M v2 MIA・T v88 AIAの予測関数 |
| `infer.py` | 既存R v1依存マニフェストの固定対象 |
| `run.py` | 同マニフェストの固定対象 |
| `run.ps1` | 同マニフェストの固定対象 |
| `test_model.py` | 同マニフェストの固定対象 |
| `dependencies.json` | 上記を含む172件の固定ハッシュ。内容を変更しない |

`../sugiyama_adapter.py` が `common.py` と `model.py` を読み込み、全依存を検証してから公開模擬入力へ予測を行います。実行に使わない4ソースも、既存の固定ハッシュゲートを弱めず維持するため必要です。その他のM/Tソース・学習済み資産・選択・過去記録は既存の正規配置を参照します。

`prepare_attack_m.ps1` の既存OneDrive互換修正もR v1の固定依存です。通常のOneDriveフォルダーを許可しつつ、実際のJunction/SymbolicLinkと管理外フォルダーは引き続き拒否し、Mソースのバイト検証を維持します。

retestの実行には `main_round/attack/retest/run.ps1` を使います。retestは `infer.py`、`run.py`、このフォルダーの `run.ps1` を呼び出さず、提出ZIPを作りません。これらに残る元のR v1提出用処理を、今回の依頼で実行する必要はありません。R v1提出CSV・提出ZIP・他のR研究成果は今回のGitHub保存対象外です。

2026-10-07のユーザー依頼はretestとその必要依存の既存非公開GitHub mainへの保存を明示的に許可しています。固定ソースの収録は公式提出や公式採点の実施を意味しません。
