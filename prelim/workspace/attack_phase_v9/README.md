# v9：最後の追加探索と guarded 候補

更新：ユーザーがv9 guardedを提出し、**最終公式得点0.3407**を共有した。
予備戦の残り提出回数は0。結果と寄与の照合は `FINAL_RESULT.md` を参照。
以下の「未提出」「条件付き下限」は提出前の研究記録であり、現在の状態ではない。

2026-09-05。4方向の新手法を実装・検証したが、採用を裏付ける安定した改善は
得られなかった。最終候補として **v9 guarded** を生成した。これは新手法の
成功を主張するものではなく、既知の得点を保全しながらv7の未得点対象向け
戦略を残すリスク調整版である。

最新の公式実測はv6の **0.3004**。v7とv9は未提出で、両者の優劣や0.5到達は
確認できていない。公式提出・pushは実施していない。残り1回は未消費。

## 最終候補と採用理由

- CSV：`submissions/prelim/team24/attack-candidate-v9-guarded/`
- ZIP：`local-output/attack-v9/attack-v9-guarded.zip`
- 生成器：`generate_guarded_submission_v9.py`
- 診断：`local-output/attack-v9/guarded_candidate.json`
- 独立監査：`local-output/attack-v9/positional/guarded_composition_audit.md`

MIAは実測v4とバイト単位で同一。AIAは次の**対象全体の326行**をコピーする。
採点結果から個々の真値を推定して行単位で継ぎ接ぎするものではない。

| コピー元 | 対象 | 根拠 |
|---|---|---|
| 実測v4 | 1, 2, 4, 5, 12, 29 | 過去の対象別最高値。2は同点 |
| 実測independent | 3, 13, 22, 23, 26 | 過去の対象別最高値 |
| 実測routed | 10, 20 | 過去の対象別最高値 |
| 未提出v7 | 6, 7, 8, 9, 17, 18, 19 | 過去3方式で全て0点。三者一致の直接選択以外に9.5を予測 |

v7との差はpositive 11対象の1,544行。全positive 13対象を実測winnerへ戻し、
未検証の9.5への時刻変更とtarget20の7件の時刻変更を外した。対象1/2と、
zero 7対象の全行はv7と同じ。これによりv7の上振れを捨てる可能性もある。
v7を上回ることを証明した変更ではない。

採点が同じ隠しデータ・仕様で行われ、保存済みフィードバックと提出物の対応が
正しい場合、次を算出できる。

```text
v4 MIA：0.2771 − mean(independent AIA) = 0.157911487923 ± 0.00005
保持するAIA：sum(13対象の実測winner) / 20 = 0.150620412542
既知成分の中心値：0.308531900465
表示丸めを考慮した条件付き下限：0.308481900465
```

各対象のAIA得点は0以上へクリップされ、対象間は平均で集約されるため、
残り7対象の未知の寄与は下限に足していない。target20の得点を二重加算して
いない。これは**新しい公式実測値でも、shadowからのLB予測でもない**。
別の評価データへの一般化を保証する下限ではなく、同じ採点集合での再利用に
依存する。0.5への到達を裏付ける根拠は得られていない。

## 新たに実装した4方向と不採用の根拠

| 実装 | 検証結果 | 判定 |
|---|---|---|
| `gen_lra.py`：局所尤度への影響 | 自己C24のheld-out 134行でTPR@1%はDCR比+0.05970。ただしAUCは0.47873、TPR差の90% block-bootstrap区間は[-0.08636, +0.09091] | 不安定で不採用 |
| `surrogate_mia.py`：公開ドナーの占有・逆セル推定 | 分離held-out 18観測のTPR差平均−0.000921、勝率27.8%。実C18/19/29の需要依存占有も固定2σゲート未達 | 不採用 |
| `registered_linkage.py`：2-view affine ICP/RANSAC | fresh 8世界のprecision平均0.7032、p10 0.5167、nullで最大14行採択。非天井18対象は全ゲート通過0 | 不採用 |
| `positional_audit.py`：行順・shuffleの漏れ | exact-label自己24のheld-out AUC 0.48180、stratified permutation FWER p=0.9081 | 不採用 |

局所尤度法は[Gen-LRAの一次論文](https://arxiv.org/abs/2508.21146)を参照し、
固定帯域Gaussian KDEに1候補を追加した際の局所リリース尤度変化を解析式で
計算した。明示的なKDE再学習との数値一致をテスト済み。論文の有効性をこの
大会へそのまま転用できるとは扱っていない。DCR/KDEとの比較であり、
v7全体に対する勝利を意味しない。

位置監査の初期診断にあったラベル依存source-index上書きとmultiplicity交絡は
発見・除去した。初期の高いAUCは無効で、採用判断に使っていない。
登録法の2モデルはanchorを作る数値列を分けているが、最終リンクには共通の
全数値列・カテゴリを使うため、完全に独立した2つの証拠ではない。自己の
イベント数も発症16件と死亡のみ6件の計22件を区別する。正しいmember時刻の
追加が0という結論は変わらない。自己targetだけの成功では他targetへ昇格させない。
単純な層別半数選択は既にv5にあるため新手法とは主張しない。

## データと検証上の制約

提供された実際のB/B_self/A_bg/schemaを明示パス・SHA-256で検証した。
練習データへの暗黙の置換はない。C19/29の公開ドナー探索で使うtoy A_bgは、
そのリリースの由来を検査するための明示的な別poolである。

探索と確認はqueryやseedを分離したが、同じB・背景・自己targetを再利用する
観測は独立な新しいデータセットとは数えない。仮定した生成方式のshadowを
他チームの真値やLB予測に読み替えない。C1/C2に完全な7-QI一致はなく、近似
対応は正解ラベルではない。自己challenge24のみ163件の完全一致を確認した。
対照群の真の時刻は不明なので、自己membership評価をAIA攻撃力とは呼ばない。
`record_id`は表内のキーで、BとA_bgの人物照合には使わない。

## 再現と安全性

```powershell
$python = "C:\Users\takut\.codex\.chatgpt-projects\g-p-6a853f607498819195a7ef70756f71c9\pwscup2026-kit-private\starter\.venv\Scripts\python.exe"
& $python -X utf8 -m attack_phase_v9.generate_guarded_submission_v9 `
  --dist "C:\Users\takut\Downloads\PWSCup2026_prelim_team24_アウトサイダー_outsider"
& $python -X utf8 -m unittest discover -v
& $python -X utf8 -m attack_phase_v5.package_submission_v5 `
  --candidate-dir .\submissions\prelim\team24\attack-candidate-v9-guarded `
  --token "C:\Users\takut\Downloads\PWSCup2026_prelim_team24_アウトサイダー_outsider\token.txt" `
  --out .\local-output\attack-v9\attack-v9-guarded.zip
```

全157テストPASS。公式logical/package validatorともPASS。生成器は入力、
feedback、probeのMIA/AIA対応、v7、manifest、出力のhashを確認し、異なる内容の
既存candidateは上書きしない。候補ディレクトリは2つのCSVのみ。ZIPは既存の
安全なpackagerだけで生成し、tokenの内容を表示せずignored領域内に保存した。
生成器の`token_read=false`はCSV生成処理の範囲を示し、その後のpackagerはtokenを
読んでいる。外部提出はしていない。

| ファイル | SHA-256 |
|---|---|
| v9 F_mia.csv | `D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B` |
| v9 F_aia.csv | `63944D5FAA2C08E5E2E40415FFBB16B3153B27EBDD052CD68D7ADB59A403A6E0` |
| v9 ZIP | `B5297F0761A88DE0C5A9B0205903138613FCD2B7A501A231AE35B391A83D0506` |

既存v7のCSV/ZIPは全てhash不変。研究の全ログ・不採用結果はignored
`local-output/attack-v9/`に保存した。
