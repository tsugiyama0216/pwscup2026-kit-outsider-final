"""A joint-epsilon14 candidate from already released DP measurements only."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import dp_copula as dc
import dp_cox as cx
from joint_cox import combine_cox
from histogram_refine import refine_models, sample_refined
from cox_optimizer import calibrate


def estimate_histogram_beta(core, public, *, n=10490, seed=2026091801, radial_strength=0):
    population=sample_refined(core,n=n,seed=seed,public_radial_strength=radial_strength)
    x=(cx._features(population)-np.asarray(public['center']))/np.asarray(public['scale'])
    frame=pd.DataFrame(x,columns=cx.COX_COLUMNS)
    frame['time']=population.time.to_numpy(); frame['onset']=population.onset.to_numpy()
    fitted=CoxPHFitter(penalizer=0).fit(frame,duration_col='time',event_col='onset')
    return fitted.params_.to_numpy(float)


def construct(models, settings):
    """No private input or evaluation callback is accepted."""
    if not 0<=settings['histogram_beta_weight']<=1:
        raise ValueError('Blend weight must lie within [0,1]')
    refined=refine_models(models,prior_multiplier=settings['histogram_prior_multiplier'])
    target=combine_cox(models,prior_multiplier=settings['cox_prior_multiplier'])
    public=models[0]['public_cox']
    if settings['histogram_beta_weight']>0:
        histogram_beta=estimate_histogram_beta(refined,public,n=settings['large_sample_n'],
            seed=settings['postprocess_seed']+1,radial_strength=settings['radial_strength'])
        w=settings['histogram_beta_weight']
        target['score_only_beta_standardized']=target['beta_standardized']
        target['histogram_beta_standardized']=histogram_beta.tolist()
        target['beta_standardized']=((1-w)*np.asarray(target['beta_standardized'])+w*histogram_beta).tolist()
        target['beta_original_units']=(np.asarray(target['beta_standardized'])/np.asarray(public['scale'])).tolist()
    target['histogram_beta_weight']=settings['histogram_beta_weight']
    original=sample_refined(refined,n=1049,seed=settings['postprocess_seed'],
                             public_radial_strength=settings['radial_strength'])
    detector=refined['schema']['rare_detector']
    strata=(dc._radii(original,detector)>=float(detector['band']['lo'])).astype(int)
    generated,trace=calibrate(original,target,public,strata=strata,
        iterations=settings['iterations'],candidates=settings['swap_candidates'],
        trials=settings['swap_trials'],seed=settings['postprocess_seed'],
        coefficient_tolerance=settings['coefficient_tolerance'])
    result={'format':'joint14_DP_postprocessing_v2','parent_epsilon_exact':'14',
        'additional_epsilon_exact':'0','delta':0,'settings':settings,
        'histogram_model':refined,'cox_target':target,'optimizer_trace':trace,
        'selection_inputs':'Public proxy selection, saved DP models and synthetic data only'}
    return generated,result,original
