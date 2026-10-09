"""Frozen Rv2 hit-probability veto; public inputs and two verified fitted assets.

Only abstention changes are allowed, on unchanged copy/independent branches.
Penalties were declared before evaluating this candidate: .75, 1, and 1.5.
The original fitted probabilities retain their historical training exposure.
"""
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
QI = ['age', 'SBP', 'TG', 'HDL', 'ALT', 'sex', 'prefecture']
PENALTIES = {'learned_075': .75, 'learned_100': 1., 'learned_150': 1.5}
FITTED_ASSETS = {
    'main_round/attack/Attak_T/local-output/attack-research/round02/models.joblib':
        '3aa172327b1b2d20d0161ecbf1e7c02efe7da89e99e2d86e0f08e725b6ba38fd',
    'main_round/attack/Attak_T/local-output/attack-research/round03/router.joblib':
        '972bd93365e642073551164b2f8248efbb0b9d920f72462b564455a23aff75e9',
}
FROZEN_SOURCES = {
    'main_round/attack/Attak_T/attack_research/features.py':
        'dc9a092b1d70b253246c851e3fe7045e6cc6134c52aac1ab7c2b9cd93c11d55c',
    'main_round/attack/Attak_T/attack_research/v2/features.py':
        '0679ba283aeda2e6559988d32319f7b03bf65e2d42f8ba940924e90b057c1210',
    'main_round/attack/Attak_T/attack_research/v3/fusion.py':
        '02d8bedc0bf202555a3b293a3e84ff4a7a71e3ffd58c357d0f295454e7bb4dcd',
    'main_round/attack/Attak_T/attack_research/v5/conditioned.py':
        'adfc7ea5cc9c6f70b0f5cb3add61a031466392c6a9c857b9d1752eb27e604745',
    'main_round/mock_test/local_eval/data.py':
        '6980de7bc4f9374bc0a4aba1d56dabb631d7306e2314dcd1d04ba9f04e4e2e84',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_dependencies():
    """Call outside a data guard, before and after inference; never reads truth."""
    for relative, expected in {**FROZEN_SOURCES, **FITTED_ASSETS}.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != expected:
            raise ValueError('Frozen learned-veto dependency differs: ' + relative)
    return {'source_sha256': dict(FROZEN_SOURCES), 'fitted_assets_sha256': dict(FITTED_ASSETS)}


def allow_assets(audit):
    """Validate both paths under the existing plugin policy before allowlisting."""
    from rank_validation.plugins import _validated_fitted_asset
    for relative, expected in FITTED_ASSETS.items():
        path = _validated_fitted_asset(relative, expected)
        if path != (ROOT / relative).resolve():
            raise ValueError('Wrong learned-veto asset root')
        audit['assets'].add(path)
        if digest(path) != expected:
            raise ValueError('Frozen learned-veto fitted asset differs: ' + relative)


def decisions(old, baseline, member_success, control_success, exact, route):
    """Apply higher penalties only when the entire .5 baseline matches exactly.

    Success arrays are joint membership-and-hit probabilities (n, proposals),
    not truth. Exact public QI matches and existing abstentions are protected.
    """
    old, baseline = np.asarray(old, float), np.asarray(baseline, float)
    ms, cs = np.asarray(member_success, float), np.asarray(control_success, float)
    exact = np.asarray(exact, bool)
    if (old.ndim != 1 or baseline.shape != old.shape or exact.shape != old.shape
            or ms.ndim != 2 or ms.shape != cs.shape or ms.shape[0] != len(old)
            or ms.shape[1] < 1 or not all(np.isfinite(x).all() for x in (old, baseline, ms, cs))
            or (old < 0).any() or (baseline < 0).any()
            or (ms < 0).any() or (cs < 0).any() or (ms > 1).any() or (cs > 1).any()):
        raise ValueError('Invalid learned-veto evidence')
    matches = bool(np.array_equal(old, baseline))
    eligible = route in ('copy', 'independent') and matches
    outputs = {}
    for name, penalty in PENALTIES.items():
        unsupported = np.max(ms - penalty * cs, axis=1) <= 0.
        remove = eligible & (old <= 10.) & ~exact & unsupported
        answer = np.where(remove, 11., old)
        outputs[name] = (answer, {'route': str(route), 'baseline_equal': matches,
            'eligible': bool(eligible), 'penalty': penalty, 'suppressed': int(remove.sum()),
            'old_active': int((old <= 10.).sum()), 'exact_links': int(exact.sum())})
    return outputs


class LearnedVeto:
    """Instantiate inside read_guard after allow_assets; _bootstrap runs before it."""
    def __init__(self):
        import joblib
        from attack_research.v2.features import extract_aia
        from attack_research.features import proposal_features
        from attack_research.v3.fusion import profile
        from attack_research.v5.conditioned import visible_conditioning
        self.extract = extract_aia
        self.proposal_features = proposal_features
        self.profile = profile
        self.condition = visible_conditioning
        loaded = {}
        for relative, expected in FITTED_ASSETS.items():
            path = ROOT / relative
            if digest(path) != expected:
                raise ValueError('Fitted asset changed before loading: ' + relative)
            loaded[Path(relative).name] = joblib.load(path)
            if digest(path) != expected:
                raise ValueError('Fitted asset changed while loading: ' + relative)
        self.model = loaded['models.joblib']
        self.router = loaded['router.joblib']

    def predict(self, release, background, challenge, old):
        old = np.asarray(old, float)
        if old.shape != (len(challenge),) or not np.isfinite(old).all() or (old < 0).any():
            raise ValueError('Invalid baseline AIA')
        for table in (release, background):
            times = table.time.to_numpy(float)
            if not np.isfinite(times).all() or np.any((times < 0.) | (times > 10.)):
                raise ValueError('Learned veto requires contest time range [0,10]')
        route = str(self.router.predict(self.profile(np, release, background)[None, :])[0])
        # Preserve every exact public QI link, including ambiguous time duplicates.
        lookup = set(release[QI].itertuples(index=False, name=None))
        exact = np.array([row in lookup for row in challenge[QI].itertuples(index=False, name=None)], bool)
        if route not in ('copy', 'independent'):
            # No extraction or speculative correction of specialized branches.
            return decisions(old, old, np.ones((len(old), 1)), np.zeros((len(old), 1)), exact, route)
        x, proposals, _, _ = self.extract(release, background, challenge)
        p = self.model['aia'].predict_proba(x)[:, 1]
        member = self.condition(challenge, p)
        hp = self.model['hit'].predict_proba(self.proposal_features(x, proposals)).reshape(len(x), proposals.shape[1], -1)
        classes = {int(k): i for i, k in enumerate(self.model['hit'].classes_)}
        if set(classes) != {0, 1, 2, 3}:
            raise ValueError('Unexpected frozen hit classes')
        mh = hp[:, :, classes[3]] / np.maximum(hp[:, :, classes[2]] + hp[:, :, classes[3]], 1e-8)
        ch = hp[:, :, classes[1]] / np.maximum(hp[:, :, classes[0]] + hp[:, :, classes[1]], 1e-8)
        ms, cs = member[:, None] * mh, (1. - member[:, None]) * ch
        gain = ms - .5 * cs
        best = gain.argmax(axis=1)
        baseline = np.where(gain[np.arange(len(x)), best] > 0., proposals[np.arange(len(x)), best].astype(float), 11.)
        return decisions(old, baseline, ms, cs, exact, route)
