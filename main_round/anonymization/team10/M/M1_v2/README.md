# M1-v2

M1を基準に、`smoking`, `time`, `onset` の割当だけを変更する候補。`death` および連続共変量・sex・prefectureは固定。

- smoking: 1/0の総数を保ったまま5%相当を交換
- time/onset: 既存M1の値ペアを行間で交換し、全体の分布を保ちながらBのCox構造と希少発症率へ較正
- death: 固定

推定本戦Utility: U=0.842778 (U_gen=0.919490, U_spec=0.842778, U_rare=0.862858, U_valid=0.874789)
DOMIAS same-20 proxy: beta=0.081154, a_MIA(K=12)=0.646853（M1と同一）
