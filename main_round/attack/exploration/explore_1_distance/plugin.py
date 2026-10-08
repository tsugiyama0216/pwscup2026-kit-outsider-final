"""Exploration 1: the same QI distance explains both attacks."""
from pathlib import Path
import json
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from explore_common.core import local_time_predictions, nearest_distance_scores, score_to_confidence

MODEL_NAME = "explore_1_distance: closest QI MIA + 1-NN time"
CONFIG = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))


def predict_world(*, candidates, background, targets, **unused_public):
    mia = pd.DataFrame({"record_id": candidates.record_id.to_numpy()})
    aia = []
    for target_id, target in sorted(targets.items()):
        print(f"Distance target {target_id}", flush=True)
        release, challenge = target["release"], target["challenge"]
        raw = nearest_distance_scores(release, candidates, background)
        mia[str(target_id)] = score_to_confidence(raw)
        times = local_time_predictions(release, challenge, background, k=1, mode="nearest")
        aia.append(pd.DataFrame({"target_k": target_id,
                                 "challenge_row_id": challenge.challenge_row_id.to_numpy(),
                                 "time_hat": times}))
    return {"F_mia": mia, "F_aia": pd.concat(aia, ignore_index=True), "diagnostics": CONFIG}
