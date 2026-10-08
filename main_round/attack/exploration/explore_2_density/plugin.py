"""Exploration 2: pinned DOMIAS MIA and a local time distribution AIA."""
from pathlib import Path
import json
import sys

import pandas as pd
from pwscup2026.attack.mia import overlap_mia

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from explore_common.core import local_time_predictions

MODEL_NAME = "explore_2_density: official DOMIAS + local time window"
CONFIG = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))


def predict_world(*, candidates, background, targets, cfg, **unused_public):
    mia = pd.DataFrame({"record_id": candidates.record_id.to_numpy()})
    aia = []
    for target_id, target in sorted(targets.items()):
        print(f"Density target {target_id}", flush=True)
        release, challenge = target["release"], target["challenge"]
        mia[str(target_id)] = overlap_mia(release, background, candidates, cfg)
        times = local_time_predictions(release, challenge, background, k=CONFIG["aia_k"], mode="window")
        aia.append(pd.DataFrame({"target_k": target_id,
                                 "challenge_row_id": challenge.challenge_row_id.to_numpy(),
                                 "time_hat": times}))
    return {"F_mia": mia, "F_aia": pd.concat(aia, ignore_index=True), "diagnostics": CONFIG}
