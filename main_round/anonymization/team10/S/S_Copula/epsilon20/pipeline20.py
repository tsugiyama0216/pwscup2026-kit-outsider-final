"""Standalone epsilon20 generator and public/DP-only postprocessing."""
from pathlib import Path
import sys
from fractions import Fraction
HERE=Path(__file__).resolve().parent;CORE=HERE.parent
sys.path[:0]=[str(CORE),str(CORE/'postprocessing_v2'),str(CORE/'expanded_budget')]
import dp_copula as dc
from influence_cox import measure
from histogram_refine import refine_models,sample_refined
from target_calibration import calibrate_target
from expanded_release import assert_preservation

def fit_once(private,public,schema,plan):
    """All newly measured queries total exactly20. No previous DP model input."""
    hist=Fraction(str(plan['histogram_epsilon']))
    if hist+sum((Fraction(str(e)) for e in plan['cox_round_epsilons']),Fraction()) != 20:
        raise ValueError('The standalone release must allocate exactly epsilon20')
    categories=sorted(str(v) for v in public.prefecture.dropna().unique())
    defaults={c:float(public[c].median()) for c in dc.NUMERIC}
    clean_public=dc._prepare(public,schema,categories,defaults)
    clean_private=dc._prepare(private,schema,categories,defaults)
    copula=dc.fit(private,public,schema,hist)
    target=measure(clean_private,clean_public,plan)
    return {'format':'S_Copula_standalone_epsilon20_v1','copula':copula,'cox_target':target,
            'accounting':{'epsilon_exact':'20','histogram_epsilon_exact':str(hist),
                'cox_epsilon_exact':target['epsilon_exact'],'delta':0,
                'adjacency':'replace_one_fixed_public_size','public_n':1049},
            'plan':plan,'previous_DP_models_used':False}

def construct(model):
    """Only DP model fields are available; no raw data or evaluation inputs."""
    plan=model['plan'];refined=refine_models([model['copula']])
    before=sample_refined(refined,seed=plan['postprocess_seed'],public_radial_strength=plan['radial_strength'])
    target=model['cox_target']
    public_target={'center':target['public_center'],'scale':target['public_scale']}
    candidate,trace=calibrate_target(before,target,public_target,model['copula']['schema']['rare_detector'],
        seed=plan['postprocess_seed'],iterations=plan['iterations'],candidates=plan['candidates'],trials=plan['trials'])
    assert_preservation(before,candidate,model['copula']['schema']['rare_detector'])
    return candidate,trace
