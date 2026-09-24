"""Standalone epsilon15 counterpart of the frozen epsilon20 mechanism."""
from pathlib import Path
from fractions import Fraction
import sys

CORE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(CORE), str(CORE / 'epsilon20')]
import dp_copula as dc
from influence_cox import measure
from pipeline20 import construct


def fit_once(private, public, schema, plan):
    hist = Fraction(str(plan['histogram_epsilon']))
    cox = sum((Fraction(str(e)) for e in plan['cox_round_epsilons']), Fraction())
    if hist != 3 or cox != 12 or hist + cox != 15:
        raise ValueError('Frozen epsilon15 allocation must be histogram3 + Cox12')
    categories = sorted(str(v) for v in public.prefecture.dropna().unique())
    defaults = {c: float(public[c].median()) for c in dc.NUMERIC}
    clean_public = dc._prepare(public, schema, categories, defaults)
    clean_private = dc._prepare(private, schema, categories, defaults)
    copula = dc.fit(private, public, schema, hist)
    target = measure(clean_private, clean_public, plan)
    return {
        'format': 'S_Copula_standalone_epsilon15_v1',
        'copula': copula, 'cox_target': target,
        'accounting': {'epsilon_exact': '15', 'histogram_epsilon_exact': '3',
                       'cox_epsilon_exact': target['epsilon_exact'], 'delta': 0,
                       'adjacency': 'replace_one_fixed_public_size', 'public_n': 1049},
        'plan': plan, 'previous_DP_models_used': False,
    }
