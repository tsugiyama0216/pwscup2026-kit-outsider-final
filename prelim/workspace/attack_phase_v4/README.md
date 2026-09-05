# PWS Cup 2026 攻撃バージョン4

`attack_phase_v4/`は、v1〜v3を変更せずに残した予備戦team24向けの独立したv4実装です。
配布済みのローカルCSVだけを読み、ネットワーク接続、token読取、提出、ZIP作成は行いません。

## 本命構成

- MIA: `v4_rrf_k10_rare_preserved`
- AIA: `v4_auc_integrity_router_v1select_horizon_t058`

### MIA

自チームのAIA challengeから特定できる163件の確定non-memberを、単なる事後降格だけでなく
target別の経験的null分布として使います。4種類の特徴viewについて候補から公開Cへの近傍距離、
相互最近傍、複数viewの同一行一致を統合し、target間の相対差も加えます。

新しいconformal順位とv3順位はRRF（`k=10`）で融合します。最後に公開希少候補内の順位をv3へ
完全復元し、163件の確定non-memberを再降格します。これにより、v2で最も強かった希少
チャネルを維持しながら全体チャネルの改善を狙います。A_bg完全一致行が多いtarget 18/29は
donor除去も比較しましたが、実指紋を再現した追加試験では元のCをそのまま使う方が強かったため、
除去版は比較variantに限定しています。

経験的較正後の主順位には同値が残るため、較正前の連続スコアを第2キー、`record_id`を最終キー
とする辞書式順位を必須にしています。

### AIA

A_bgだけで公開QIからeventを予測するモデルを学習し、各CでQIとeventの対応が保たれているかを
ROC AUCで測ります。AUCが非有限または厳密に`0.58`未満のtargetだけ、v3と同じactive setを
維持したままactive行の`time_hat`を10.0へ置換します。実データで切り替わるのは、event率100%
のtarget 17だけです。density型horizon候補への切替はmembership選択まで変わるため使いません。

## 実行

```powershell
powershell -ExecutionPolicy Bypass -File .\attack_phase_v4\run_v4.ps1 `
    -Dist ..\pwscup2026-kit\work\prelim\team24\runtime_dist `
    -Targets .\team-shared\prelim\attack-targets `
    -OutputDir .\local-output\prelim\team24\attack-phase-v4 `
    -CandidateDir .\submissions\prelim\team24\attack-candidate-v4
```

`CandidateDir`には本命の`F_mia.csv`と`F_aia.csv`だけを生成します。`OutputDir`には比較variantと
`diagnostics_v4.json`も保存します。`-SkipDomias`は高速な動作確認専用で、本命生成には使いません。

## 生成時の検査

- target集合がteam24用の20件と完全一致
- self challengeの326行中、厳密に半数の163行を確定non-memberとして一意に照合
- MIAは1,049行、公式列集合、record ID一意、有限`[0,1]`、全target tie-free
- AIAは6,520行、全challenge複合キー一意、有限・非負、active set不変
- 公式キットのMIA/AIA論理validator
- candidate CSVを再読込して同じ検査を再実行

詳細な比較根拠は`EXPERIMENT_RESULTS.md`に記録しています。
