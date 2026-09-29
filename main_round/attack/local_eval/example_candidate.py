"""Working prediction adapter example. Replace the body with your attack."""
from local_eval.methods import multiview


def predict(*, c, background, candidates, challenge, cfg):
    # No truth labels, source row identities, or case-selection seed are supplied.
    return multiview(c=c, background=background, candidates=candidates,
                     challenge=challenge, cfg=cfg)
