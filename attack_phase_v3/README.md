# PWS Cup 2026 攻撃バージョン3

`attack_phase_v3/`は、`attack_phase/`（v1）と`attack_phase_v2/`を変更せずに残した、独立した
v3実装です。配布済みのローカルCSVだけを読み、ネットワーク接続・token読取・提出操作は行いません。

## v2実測からの方針

v2のリーダーボード結果は、合計`0.28`、MIA全体`0.0867`、MIA希少`0.3333`、
AIA全体`0.1171`、AIA希少`0.1000`でした。合計は4内訳の単純和ではなく、各targetで
MIA/AIAそれぞれ`max(全体, 希少)`を取ってから集約されます。

v3は、MIA全体の低FPR順位に残っていた確定負例を除くことを優先します。AIAの追加案は
self既知ラベルとshadow stressで比較しましたが、現行v2を安定して上回らなかったため、本命は
v2 AIAを維持します。

## MIA本命: self challenge確定負例の降格

ルール上、自チーム向けAIA challengeのmember側163行は、他チームと共有されない自チームの
非共有ブロックから作られます。予備戦team24では、`aia_challenge_24.csv`と`B.csv`を公開QI
7列（`age, SBP, TG, HDL, ALT, sex, prefecture`）で完全一致すると、設計どおり163行を一意に
特定できます。この163行は、他の全20targetに対する確定non-memberです。

v3本命`v3_private_demotion_transport`は、v2のDOMIAS込みsoft transportを計算した後、確定負例
だけを各target列の最下位へ移し、残り886行の相対順位を完全に維持します。正例を一切降格せず、
確定負例だけを下げるため、各targetのROC/TPR@FPRを悪化させない変換です。

実データでは、v2の上位10/20/50枠に確定負例が全target合計で`14/33/134`件ありました。
v3ではすべて`0`件です。また、希少24行のうち13行が確定負例で、v2では20target中8targetの
希少top-1を占めていましたが、v3では0targetになりました。v3の全200 top-10枠は引き続き
200人に分散し、soft transportの標的横断排他性も維持しています。

比較用として、確定負例を最初から除いて886行でtransportを再計算する方式、希少補正方式、
v1 unconstrained順位を10%混ぜる方式も`variants/`へ出します。本命は、実戦真値がなくても
非悪化を保証できるpost-hoc降格です。

## AIA本命: v2を維持

self target 24の既知memberで比較すると、50%会員選別はv1がTP84/FP79、実験的な密度選別が
TP85/FP78で、差は163人中1人だけでした。時刻復元のmember成功率はv1`0.4356`からv2
`0.4479`へ改善しました。一方、選択率0.40への変更は3 stress条件の公式shadow平均が
`0.8352`から`0.7153`へ低下し、希少補正の最高も`0.8308`でした。

このためv3の`F_aia.csv`はv2本命と同一です。公開Cのevent率が100%のtarget 17だけを
horizon-onlyにする案は`v3_anomalous_horizon_experimental`として比較出力へ残しますが、
本命には採用しません。

## 実行

予備戦team24の再現コマンドです。

```powershell
powershell -ExecutionPolicy Bypass -File .\attack_phase_v3\run_v3.ps1 `
    -Dist ..\pwscup2026-kit\work\prelim\team24\runtime_dist `
    -Targets .\team-shared\prelim\attack-targets `
    -SelfTargetId 24 `
    -ExcludeTargets 24 `
    -OutputDir .\local-output\prelim\team24\attack-phase-v3 `
    -CandidateDir .\submissions\prelim\team24\attack-candidate-v3 `
    -MiaTotalShared 508
```

`CandidateDir`には本命の`F_mia.csv`と`F_aia.csv`だけを生成します。`OutputDir`には比較variantと
`diagnostics_v3.json`も保存します。`--skip-domias`相当の`-SkipDomias`は高速な動作確認用であり、
本命候補には指定しません。

## 安全検査

- self challenge照合は、偶数326行、公開QI複合キー一意、候補一致が厳密に半数163行、
  record ID一意を満たさなければ生成を停止
- MIAはID集合、対象列、有限`[0,1]`、targetごとのtie-freeを検査
- AIAは全target/challenge ID、複合キー一意、有限非負時刻を検査
- 公式キットのMIA/AIA論理validatorにも通す

提出ZIPを作る場合はREADMEや`variants/`を含めず、直下にこの2枚とユーザー管理の
`token.txt`だけを置いてください。
