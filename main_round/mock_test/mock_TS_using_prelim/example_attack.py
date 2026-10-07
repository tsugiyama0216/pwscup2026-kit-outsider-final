"""Runnable official baseline and template for user-supplied attacks."""
from __future__ import annotations

import pandas as pd
from pwscup2026.attack.aia import aia_predict
from pwscup2026.attack.mia import overlap_mia

MODEL_NAME = "公式参照攻撃（予備戦データ）"


def predict_world(*, candidates, self_labels, background, self_challenge,
                  targets, schema, cfg, attacker_id=24, self_release=None,
                  utility_reference=None):
    """Use the provided actual inputs. Do not read the separate ground truth."""
    mia = pd.DataFrame({"record_id": candidates.record_id.to_numpy()})
    aia = []
    for k, target in sorted(targets.items()):
        print(f"Predicting target {k}", flush=True)
        mia[str(k)] = overlap_mia(target["release"], background, candidates, cfg)
        _, time_hat = aia_predict(target["release"], target["challenge"], background, cfg)
        aia.append(pd.DataFrame({"target_k": k,
                                 "challenge_row_id": target["challenge"].challenge_row_id.to_numpy(),
                                 "time_hat": time_hat}))
    return {"F_mia": mia, "F_aia": pd.concat(aia, ignore_index=True)}
