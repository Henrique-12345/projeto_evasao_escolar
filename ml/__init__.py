"""Pipeline de machine learning — evasão escolar (Recife)."""

from ml.educational_ml import (
    load_final_model_bundle,
    predict_taxa_abandono_em,
    run_educational_ml_suite,
)
from ml.scenario_simulation import (
    SIMULATABLE_FEATURES,
    build_intervention_narrative,
    predict_abandono_row,
)

__all__ = [
    "run_educational_ml_suite",
    "predict_taxa_abandono_em",
    "load_final_model_bundle",
    "SIMULATABLE_FEATURES",
    "build_intervention_narrative",
    "predict_abandono_row",
]
