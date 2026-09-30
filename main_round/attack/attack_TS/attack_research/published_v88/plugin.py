"""The frozen v88 method in the local evaluator's public-input API."""
from __future__ import annotations

from functools import lru_cache

from attack_research.v88.predict import CombinedAttack
from attack_research.published_v88.check import verify_package


@lru_cache(maxsize=1)
def _attack() -> CombinedAttack:
    verify_package()
    return CombinedAttack()


def predict(*, c, background, candidates, challenge, cfg):
    answers, _ = _attack().predict(c=c, background=background,
                                   candidates=candidates, challenge=challenge, cfg=cfg)
    return answers['combined']
