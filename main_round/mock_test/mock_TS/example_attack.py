"""Copy this interface into your attack's own workspace.

Only supplied public inputs may be used. Do not load original B, full A_bg,
truth tables, prior predictions, or fitted artifacts trained on held-out people.
"""
from pwscup2026.attack.aia import aia_predict
from pwscup2026.attack.mia import overlap_mia


def predict(*, c, background, candidates, challenge, cfg):
    """Each invocation is a fresh single-C process; preserve input row order."""
    _, time_hat = aia_predict(c, challenge, background, cfg)
    return {'mia': overlap_mia(c, background, candidates, cfg), 'aia': time_hat}
