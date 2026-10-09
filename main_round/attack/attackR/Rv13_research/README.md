# Rv13〜Rv15：独立した攻撃方式の検討（2026-10-09）

**依頼条件「2種類の自己採点でRv2を超える」は未達。3候補ともRv2の置き換えには未採用です。**
Rv2/Rv8のコード・予測・学習済み資産を推論に使用せず、公開されているB、A_bg、各C、challengeから計算する新規実装を作成しました。11系統の方式を試し、性質の異なる3候補を保存しています。採点関数は変更していません。

## 確定比較

| モデル | 予備戦実データ | 本戦模擬・開発2ケース平均 | 本戦模擬・追加確認2ケース平均 | 両方でRv2超え |
|---|---:|---:|---:|---|
| Rv2 | 0.401220092 | 0.886143599 | 1.002980985 | 基準 |
| Rv13 | 0.249542687 | 1.008744085 | 1.073239558 | 未達 |
| Rv14 | 0.242064079 | 0.998928571 | 1.052719179 | 未達 |
| Rv15 | 0.339814503 | 0.677903866 | 0.761548137 | 未達 |

高い方が良い値です。予備戦と本戦模擬は異なる指標なので、列をまたいだ値の大小や合算は意味しません。

- 予備戦：先ほどの `mock_TS_using_prelim` による20対象の攻撃力（MIA power + AIA power）。公開済み正解を採点だけに使用。本戦のAIA floorは適用しません。
- 本戦模擬：先ほどのv5指標 `stable_index = 対象ごとの (mia_all + 補正後aia_power) の平均`。22対象。開発seedは2026100961/62、追加確認は2026100981/82。実際の本戦の公式得点ではありません。
- 設定の選択には予備戦評価と開発2ケースを使用しました。追加確認の前にモデルのSHA256を固定し、確認後には調整していません。ただし追加ケースも同じ生成系・母集団であり、別分布への汎化の証明ではありません。
- Rv13/Rv14は本戦模擬の改善を追加ケースでも確認できましたが、予備戦のMIA/AIAがともに不足しています。Rv15は予備戦MIAがRv2を少し上回る一方、AIAの低下を補えませんでした。

## 保存した方式

| モデル | MIA：含まれる人物の推定 | AIA：timeの推定 |
|---|---|---|
| [Rv13](../Rv13/README.md) | 8数値列の最近傍C距離と背景5近傍距離の比 | 背景共分散で補正したQI距離＋弱いカテゴリ罰則、最近傍time |
| [Rv14](../Rv14/README.md) | Rv13と同じ | QI数値列の標準化距離、最近傍time |
| [Rv15](../Rv15/README.md) | 公開背景を分割した経験的帰無分布で距離比を校正 | QI＋カテゴリ距離、背景がより近い行ではtime=11に抑制 |

3モデルとも、自分の公開challengeとの厳密な重なりから確認できる163人を他対象の非メンバーとして扱います。同距離のAIA候補はtimeを平均し、行順で結果が変わらないようにしています。学習済み資産は不要です。

## 再現と検証

クローンしたリポジトリのルートから実行してください。初回は `main_round/attack/setup.ps1` で固定環境を準備します。このPCでは既存の環境を指定しても実行できます。`--model`はRv13/Rv14/Rv15から選択し、出力先は毎回新しい名前にします。

```powershell
$py = (Resolve-Path './main_round/attack/.venv/Scripts/python.exe').Path
& $py -B main_round/attack/attackR/Rv13_research/run_model.py --model Rv13 --output main_round/mock_test/local-output/rv13-reproduce-new
```

各モデルの `predictions/F_mia.csv` と `predictions/F_aia.csv` は、本戦の実際の公開入力から生成したローカル候補です。真値による本戦得点は不明です。2回の別プロセス生成でCSVのバイト一致を確認しました。形式検証、50公開入力の不変、元94項目のmanifest検証、7件の契約テストを実施しました。監査は信頼するローカルPythonでのファイル読取監査であり、OS全体の隔離を証明するものではありません。

新旧モデルの公開入力、採点記録とモデルSHA256を記録しています。既存の追跡ファイルと凍結ゲートを変更していません。初回検証時はGitHubへのpush、提出ZIP作成、CodaBench提出を行っていません。続くユーザーの依頼により、固定モデル・本戦CSV・選択と検証記録・この比較資料・実行とテストの入口を既存の非公開GitHubのmainへ保存する対象にしました。保存済みJSONの `github_push_performed: false` は初回検証時の履歴として維持します。

## 記録

- [比較CSV](comparison.csv)、[ケース別CSV](case_scores.csv)、[比較定義・対象別感度・証跡JSON](comparison.json)
- [11系統の探索集計](search_summary.csv)。成績の良い設定だけでなく、実行した各系統の集計先とハッシュを記載しています。
- [テストログ](evidence/tests.log)。採点JSONと対象別CSVも `evidence/` に保存。
- 全試行の推論・採点出力：`main_round/mock_test/local-output/rv13-research-20261009/`。失敗した初期準備も履歴として残しています。
- 探索時の一部provenanceにある旧キー `all_25_combinations_retained` は固定文言でした。実際の組合せ数は各 `summary.csv` の行数および `search_summary.csv` の `component_pairs` を参照してください。保存済みの証跡は書き換えていません。

予備戦での不足を解決したとは扱えません。今回の証拠からは、Rv2を置き換える根拠はありません。

## GitHub保存時の確認

最新mainへ重ねた状態で3モデルの本戦予測を再生成し、固定CSVとの完全一致を照合済みです。`test_models.py` はGitに含まれる公開本戦入力を小さく切り出して動作し、ローカル限定の模擬ケースを必要としません。モデル・予測・選択・過去の検証記録は変更しません。

```powershell
& $py -B main_round/attack/attackR/Rv13_research/test_models.py
& $py -B main_round/attack/attackR/Rv13_research/verify_saved.py
```

探索エンジンと全試行の出力は元の作業用worktreeに残し、今回は完成候補の実行に必要なファイルと選択済み証跡だけを保存します。全探索の再実行用パッケージではありません。元の絶対パスを含む評価JSONは当時の履歴であり、推論時には読みません。生成ケース・正解・作業中の予測・環境・キャッシュ・認証情報・ZIPは新たに含めません。

GitHub保存時の結果は [publication_verification.json](publication_verification.json)、今回保存する研究補助ファイルのハッシュは [publication_manifest.json](publication_manifest.json) に記録します。
