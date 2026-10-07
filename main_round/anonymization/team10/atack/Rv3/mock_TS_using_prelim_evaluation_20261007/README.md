# Rv3の追加MOCK再評価（2026-10-07）

ユーザーの再評価依頼により、保存済みRv3を新しい `mock_TS_using_prelim` で評価しました。**攻撃力は0.121774285で、目標0.4には未達**です。Rv3の選択・方式・パラメーターを変更せず、予備戦team24の公開入力と20対象に接続するアダプターだけを追加しました。

| 指標 | 得点 |
| --- | ---: |
| 攻撃力合計 | 0.12177428492165779 |
| MIA力 | 0.06784615384615385 |
| AIA力 | 0.05392813107550394 |

[対象別のHTMLレポート](report.html)、[採点結果JSON](result.json)、[対象別CSV](by_target.csv)、[検証記録](verification.json)を保存しています。

## 今回の評価条件

- 人工生成した問題ではなく、予備戦の実配布データと運営公開正解を使用。公開済みデータに対する事後評価であり、新しい公式提出ではありません。
- 攻撃者は予備戦team24、自己24を除く20対象。MIAは1,049候補、AIAは20対象×326問＝6,520行です。本戦team10の22対象・提出CSVとは別の予測です。
- 予備戦公式実装 `prelim-attack-20260825` で採点。本戦の対象別AIA補正値φは適用せず、予備戦の希少AIAに対する1.64×SE控除を適用します。
- 各対象で全体・希少の大きい値を採り、その後20対象の平均を計算します。MIA力とAIA力の和が攻撃力です。内訳の `mia_rare=0.166666667` は希少正例が存在する6対象だけの平均であり、MIA力そのものではありません。
- 以前のmock_TS確認平均0.063132729、杉山22対象共同評価0.071870131とは入力・採点条件が異なります。差を改善量と解釈せず、本戦の得点予測にも用いません。**本戦Rv3の公式得点は未測定のままです。**

MIAは12/20対象、AIAは11/20対象で得点0でした。対象5の希少MIA得点1が合計に0.05寄与しています。これらは今回の内訳であり、方式の調整や選び直しには使っていません。

## 凍結と再現性

MIA `full5` とAIA `forest_contrast5_1.0` を凍結したまま実行しました。本番と同じ公開情報だけの非共有者判定を予備戦入力に適用し、163人のMIA確信度を0にしています。旧Tの学習済みモデルや保存ドナーは使っていません。攻撃側に与えるのは配布公開表だけで、対象正解や他チームの回答を引数に渡していません。

独立した新規プロセスで2回生成・公式実装による採点を行い、両CSVのバイト列と全20対象の得点が完全一致しました。48件の既存依存、Rv3の凍結選択、本戦提出CSV2件とZIPのハッシュも保存済み記録と一致しました。元の提出データは変更していません。

`adapter.py` は推論中のCSV・モデル資産等のファイル読込みをPython監査で拒否し、試行は0件でした。これは信頼したローカルソースの確認と拡張子ベースの監査であり、OSのファイル隔離を意味しません。正解は評価器だけで使用します。

入力束の178ファイルと予備戦公式ソース36ファイルは、評価器に同梱されたmanifestと照合済みです。使用した評価器はGitHub mainに追加済みの `main_round/mock_test/mock_TS_using_prelim/`（初出コミット `335e695`）です。アダプター・評価器・公開入力・予測のハッシュは `prediction_provenance.json`、元モデルのハッシュは `plugin_metadata.json` と `verification.json` に記録しています。

## 再実行

リポジトリ直下で、本戦の既存Python環境を使用します。各出力先は存在しない新しいディレクトリを指定してください。

```powershell
main_round/attack/.venv/Scripts/python.exe -I -B main_round/mock_test/mock_TS_using_prelim/cli.py Run --plugin main_round/anonymization/team10/atack/Rv3/mock_TS_using_prelim_evaluation_20261007/adapter.py --output main_round/mock_test/local-output/mock_TS_using_prelim/Rv3_20261007
main_round/attack/.venv/Scripts/python.exe -I -B main_round/mock_test/mock_TS_using_prelim/cli.py Run --plugin main_round/anonymization/team10/atack/Rv3/mock_TS_using_prelim_evaluation_20261007/adapter.py --output main_round/mock_test/local-output/mock_TS_using_prelim/Rv3_20261007_repeat
main_round/attack/.venv/Scripts/python.exe -I -B main_round/anonymization/team10/atack/Rv3/mock_TS_using_prelim_evaluation_20261007/verify.py
```

`verify.py` は保存済み2回の出力を照合し、このフォルダーのレポートと検証記録を作ります。既存の本戦提出ZIPを保ったまま検証するため、ZIPがない新規cloneでは本戦提出生成も必要です。トークン入りZIPは従来どおりPCのみ、Git除外です。評価用の予測CSV・入力コピー・ログは共有の無視対象出力先に保存し、GitHubにはこのアダプターと非認証の集計・検証記録だけを追加します。公式提出・公式試行の消費はしていません。
