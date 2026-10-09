# 完成した6候補の比較

得点欄はすべて未測定です。下表は本戦公開入力でRv2からどのくらい予測が変わったかを示すもので、正解数・得点差ではありません。

| モデル | MIA順位変更数 | 上位30件の入替数 | AIA変更数 | 0.5年を超える時刻変更数 | 公式・新採点 |
|---|---:|---:|---:|---:|---|
| Rv6 | 19,284 | 159 | 0 | 0 | 未測定 |
| Rv7 | 19,320 | 175 | 0 | 0 | 未測定 |
| Rv8 | 19,105 | 99 | 0 | 0 | 未測定 |
| Rv9 | 0 | 0 | 600 | 496 | 未測定 |
| Rv10 | 0 | 0 | 854 | 756 | 未測定 |
| Rv11 | 19,200 | 118 | 781 | 652 | 未測定 |

MIA順位変更の母数は既知陰性163件を除いた886候補×22対象=19,492です。上位30件は22対象合計660枠で、同じ候補でも別対象なら別に数えます。AIAの母数は7,172回答です。対象別の原値は各版の `production_diagnostics.json` にあります。

Rv6/Rv7/Rv8のAIA、Rv9/Rv10のMIAは、保存済みRv2のCSVとバイト単位で一致します。6版の予測CSVペアはすべて互いに異なります。

検証は全版で、2回の独立本番生成の一致、CSV再読込、公式形式・実ID・ZIP検査、8種類の不正提出の拒否が成功しました。公開入力の数学的性質など9件のテストも成功しています。最新mainを元にした新しい作業フォルダーで、126依存ファイル、全版の提出CSV、9件のテストを再確認しました。別の物理PCでの実行は未確認です。

特にAIAの「公開Cの時刻分布と背景の差」は、真のメンバー正解率とコントロール正解率の差そのものではありません。時刻加工が強いCでは誤った追加信号にもなり得ます。Rv9/Rv10/Rv11の変更数が多いことを、そのまま性能向上とは解釈しないでください。

## 保存先

- PCの基点：`C:/Users/Owner/OneDrive/ドキュメント/ChatGPT/pws2026abe2/main_round/anonymization/team10/attack/attackR/`
- GitHubの基点：<https://github.com/tsugiyama0216/pwscup2026-kit-private/tree/main/main_round/anonymization/team10/attack/attackR>
- 各版：`RvN/submission/F_mia.csv`、`RvN/submission/F_aia.csv`
- PC提出ZIP：`RvN/submission/team10_RvN_attack.zip`（N=6〜11、認証情報を含むためGit除外）

GitHubからのZIP再作成は `assemble_saved_zip.py` を使います。新採点との接続は [EVALUATOR_HANDOFF.md](EVALUATOR_HANDOFF.md)、全方式の説明と未実装案は [README.md](README.md)、[RESEARCH_BACKLOG.md](RESEARCH_BACKLOG.md) を参照してください。
