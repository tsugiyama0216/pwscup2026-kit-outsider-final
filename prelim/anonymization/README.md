# 予備戦・匿名化フェーズの提出物

[`anonymization-submitted-v5/C.csv`](anonymization-submitted-v5/C.csv) は、team24 が予備戦の**匿名化フェーズ**で提出した匿名化データです。攻撃フェーズの v5 候補ではありません。

旧 `submissions/prelim/team24/submitted-v5/` を、このディレクトリの `anonymization-submitted-v5/` に移動・改名しました。提出済み `C.csv` の内容は保持しています。

```text
anonymization/
├── README.md
└── anonymization-submitted-v5/
    ├── .gitattributes
    └── C.csv
```

匿名化の実装・設定は [`../workspace/opendp_aim/`](../workspace/opendp_aim/README.md)、当時のルールは [`../workspace/rulebook/`](../workspace/rulebook/) を参照してください。再実行は `prelim/workspace/` を起点にし、既存の提出物を不用意に上書きしないでください。

実トークンとトークン入り ZIP は新たなアップロードに含めません。既存のローカルファイルは保持し、過去の Git 履歴も書き換えていません。攻撃提出物は別の [`../attack/`](../attack/README.md) にあります。
