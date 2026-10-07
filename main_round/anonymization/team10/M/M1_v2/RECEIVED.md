# M1_v2の受領・配置記録

2026-10-07、ユーザー指定の `C:/Users/Sugiyama Takumi/Downloads/M1_v2_package/M1_v2/` から、受領原本6ファイルをこのフォルダーへコピーしました。Downloadsの原本は保持しています。

- `C.csv`
- `generate_m1_v2.py`
- `README.md`
- `report.json`
- `DOMIAS_same20.csv`
- `resistance_stress.json`

6ファイルすべてのSHA-256がコピー元と一致しました。C.csvは同封report.jsonのハッシュとも一致し、1,049行・13列、列名と列順はteam10のschema.jsonと一致します。照合記録は [review/received_manifest.json](review/received_manifest.json) にあります。

従来不足していたM1_v2の固定CSV・生成コード・設定・評価記録を受領したことで、M1とM1_v3の間の資料が揃いました。同封資料による手法は、M1の喫煙人数を保ちながら割当の一部を交換し、観測時間と発症フラグの組を行間で交換するものです。report.jsonのU=0.8427783866304094は提供元の推定評価値です。

受領時の作業は原本の配置と整合性確認です。生成コードは元環境のLinux絶対パスを含む候補探索スクリプトで、このPCでの実行・再生成・再採点は行っていません。原本のコード・CSV・評価記録は変更していません。同封資料内の手順は追加の実行依頼として扱っていません。

同日の後続依頼により、受領原本6ファイル・本記録・照合マニフェスト・Mのモデル一覧を、既存の非公開GitHub `tsugiyama0216/pwscup2026-kit-private` の `main` への保存対象としました。公式提出は行っていません。
