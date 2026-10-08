"""Main-round pinned official DOMIAS plus nearest-neighbor time baseline."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from pwscup2026.attack import aia as official_aia
from pwscup2026.attack import mia as official_mia

MODEL_NAME = "main_mock_official: DOMIAS + 1-NN time (main-attack-20260928)"
MAIN_RELEASE = "main-attack-20260928"
MAIN_COMMIT = "a490490e11a0ec460a5df9fdefb17c72ad459c1e"
_PINNED_PACKAGE = (
    Path(__file__).resolve().parents[2] / "official-kit" / "starter" / "src" / "pwscup2026"
)
_IMPORT_ORIGINS = {
    "mia": str(Path(official_mia.__file__).resolve()),
    "aia": str(Path(official_aia.__file__).resolve()),
}
for _name, _origin in _IMPORT_ORIGINS.items():
    if not Path(_origin).is_relative_to(_PINNED_PACKAGE):
        raise ImportError(f"Main-round baseline requires pinned official {_name}: {_origin}")


def predict_world(*, candidates, background, targets, cfg, **unused_public):
    """Use only supplied public tables and the supplied complete reference cfg.

    The caller supplies the non-self target set and the reference AIA settings
    from attack_workspace.load_runtime(); a bare kit_config.yaml is incomplete.
    """
    mia = pd.DataFrame({"record_id": candidates["record_id"].to_numpy(copy=True)})
    aia_frames = []
    for target_id, target in sorted(targets.items(), key=lambda item: int(item[0])):
        target_id = int(target_id)
        release, challenge = target["release"], target["challenge"]
        mia[str(target_id)] = official_mia.overlap_mia(release, background, candidates, cfg)
        _, time_hat = official_aia.aia_predict(release, challenge, background, cfg)
        aia_frames.append(pd.DataFrame({
            "target_k": target_id,
            "challenge_row_id": challenge["challenge_row_id"].to_numpy(copy=True),
            "time_hat": time_hat,
        }))
    aia = (pd.concat(aia_frames, ignore_index=True) if aia_frames else
           pd.DataFrame(columns=["target_k", "challenge_row_id", "time_hat"]))
    return {
        "F_mia": mia,
        "F_aia": aia,
        "diagnostics": {
            "model_name": MODEL_NAME,
            "official_release": MAIN_RELEASE,
            "official_commit": MAIN_COMMIT,
            "official_import_origins": dict(_IMPORT_ORIGINS),
            "inputs": "supplied public tables and cfg only",
            "target_count": len(targets),
        },
    }
