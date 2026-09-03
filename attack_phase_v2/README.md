# PWS Cup 2026 攻撃バージョン2

`attack_phase_v2/`は、従来の`attack_phase/`と提出候補v1を変更せずに比較・改善する独立実装です。
ネットワークアクセスや提出操作は行わず、配布済みCSVだけを読みます。

## 現状分析と採用判断

公式得点は `MIA power + AIA power` です。事務局限定の正解は手元にないため、既知の疑似真値を
使い、キットの公式採点関数でv1/v2を同条件比較しました。

- MIA: v1の標的横断soft transportが強く、追加した候補局所kNN密度比と`B_self.is_rare`を
  使う希少行transportはshadow評価で下回りました。v2本命はv1 MIAを維持し、新方式は
  `variants/`に比較候補として残します。
- AIA: v1の会員選別は維持します。非選択行を真値範囲内の`0.0`年に置くと偶然的中するため、
  真値範囲を外す有限sentinel（既定`11.0`年）へ変更します。選択行も、対応C行が打切りなら
  公開horizon、イベント行でもリンク距離が弱ければhorizonへ縮約し、十分近いイベント行だけ
  C行の時刻を使います。通常の信頼距離はshadowで選んだ`0.20`です。ただし公開Cのイベント率が
  50%を超える異常な対象は`0.08`へ自動的に戻し、壊れた転帰列を過信しません。

つまり、v2本命は「MIAと会員選別を壊さず、AIAの時刻復元だけを改善」した保守的な候補です。

## 実行

練習データで全比較候補と診断を生成します。

```powershell
powershell -ExecutionPolicy Bypass -File .\attack_phase_v2\run_v2.ps1 `
    -Dist .\participant_data `
    -OutputDir .\local-output\attack-phase-v2
```

予備戦team24の再現例です。自チーム`24`を除外し、20対象の共有総数事前を508にします。
`CandidateDir`には本命CSV 2枚だけを同時に書き出します。

```powershell
powershell -ExecutionPolicy Bypass -File .\attack_phase_v2\run_v2.ps1 `
    -Dist ..\pwscup2026-kit\work\prelim\team24\runtime_dist `
    -Targets .\team-shared\prelim\attack-targets `
    -ExcludeTargets 24 `
    -OutputDir .\local-output\prelim\team24\attack-phase-v2 `
    -CandidateDir .\submissions\prelim\team24\attack-candidate-v2 `
    -MiaTotalShared 508
```

`--skip-domias`相当の`-SkipDomias`は動作確認向けです。本命候補には指定しません。

## 回帰評価

```powershell
powershell -ExecutionPolicy Bypass -File .\attack_phase_v2\evaluate_v2.ps1 -SkipDomias
```

既定では2 seed × 5条件（無加工、3%/10%摂動、時刻シャッフル、独立負例）を評価します。
出力は`local-output/attack-shadow-evaluation-v2/`です。これは方式間の回帰比較であり、
公式得点や順位の推定ではありません。

今回の採用比較と判断根拠は`EXPERIMENT_RESULTS.md`に固定しています。

## 出力と安全確認

- `F_mia.csv`, `F_aia.csv`: v2本命
- `variants/`: 採否を比較するMIA/AIA候補
- `diagnostics_v2.json`: 集計診断（個票ラベルは保存しません）
- 生成前検査: MIAのID完全一致・有限`[0,1]`・全列strict rank、AIAの全challenge ID完全一致・
  複合キー一意・時刻の有限非負を確認

プログラムは`token.txt`を読み書きしません。提出ZIPを作る場合はREADMEや`variants/`を含めず、
直下に`F_mia.csv`、`F_aia.csv`、ユーザー管理の`token.txt`だけを置いて公式validatorを通します。
