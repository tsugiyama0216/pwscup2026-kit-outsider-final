"""Public-table plugin for the common PWSCUP evaluation API."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ATTACK_ROOT = Path(__file__).resolve().parent.parent
if str(_ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_ATTACK_ROOT))

from explore_3_gen_lra.model import FixedKernelGenLRA
from explore_common.core import local_time_predictions, score_to_confidence

MODEL_NAME = "explore_3_gen_lra_fixed_kernel_k10"

# Mirrored in config.json; inference performs no filesystem reads.
CONFIG = {'model_name': 'explore_3_gen_lra_fixed_kernel_k10',
 'version': 1,
 'selection': 'fixed_before_preliminary_evaluation',
 'mia': {'method': 'fixed_kernel_Gen_LRA',
         'features': ['age', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'FPG'],
         'reference': 'supplied_background',
         'k': 10,
         'neighborhood': 'Euclidean_after_reference_standardization',
         'missing_values': 'reference_column_median',
         'standardization': 'reference_mean_and_sample_std',
         'covariance': 'reference_sample_covariance_after_standardization',
         'minimum_covariance_eigenvalue': 1e-06,
         'bandwidth': 'Silverman_factor_(m*(d+2)/4)^(-1/(d+4))',
         'augmented_kde': 'reuse_reference_covariance_and_bandwidth',
         'raw_score': 'log(sum_s softplus(log_K_H(s-x)-log(m*p_R(s))))',
         'confidence': 'explore_common.core.score_to_confidence',
         'batch_size': 256,
         'distance_tie_policy': 'lexicographic_standardized_seven_features_fixed_k',
         'numerical_variant': 'monotone_log_positive_evidence_no_density_floor_no_subtractive_baseline'},
 'aia': {'method': 'explore_common.core.local_time_predictions', 'k': 10, 'mode': 'window'},
 'reads': 'supplied_public_tables_only',
 'not_used': ['target_truth',
              'public_answer_archives',
              'self_labels',
              'self_challenge',
              'self_release',
              'utility_reference']}


def predict_world(*, candidates, self_labels, background, self_challenge,
                  targets, schema, cfg, attacker_id=24, self_release=None,
                  utility_reference=None):
    """Infer only from supplied public tables, with fixed, label-free settings.

    self_labels, self_challenge, self_release, schema, cfg and utility_reference
    are deliberately unused. No answer file or other team's truth is read.
    """
    model = FixedKernelGenLRA(background, k=10)
    mia = pd.DataFrame({"record_id": candidates.record_id.to_numpy()})
    aia = []
    for target_k, target in sorted(targets.items(), key=lambda pair: int(pair[0])):
        print(f"{MODEL_NAME}: predicting target {target_k}", flush=True)
        release, challenge = target["release"], target["challenge"]
        raw = model.score(release, candidates)
        mia[str(target_k)] = score_to_confidence(raw)
        time_hat = local_time_predictions(
            release, challenge, background, k=10, mode="window"
        )
        aia.append(pd.DataFrame({
            "target_k": target_k,
            "challenge_row_id": challenge.challenge_row_id.to_numpy(),
            "time_hat": time_hat,
        }))
    f_aia = (pd.concat(aia, ignore_index=True) if aia else
             pd.DataFrame(columns=["target_k", "challenge_row_id", "time_hat"]))
    return {
        "F_mia": mia, "F_aia": f_aia,
        "diagnostics": {
            "fixed_config": CONFIG,
            "reference_rows": model.n_reference,
            "dimension": model.n_features,
            "silverman_factor": float(model.bandwidth),
            "kernel_variant": "reference_fixed_covariance_and_bandwidth_no_augmented_refit",
            "raw_score_variant": "log_sum_softplus_monotone_in_fixed_kernel_GenLRA",
        },
    }
