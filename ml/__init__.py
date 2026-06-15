"""Pipeline de machine learning — evasão escolar (Recife)."""

from ml.educational_ml import (
    load_final_model_bundle,
    predict_taxa_abandono_em,
    run_educational_ml_suite,
)
from ml.mlflow_experiments import (
    EXPERIMENT_PLAN,
    HGB_EXPERIMENT_GRID,
    HGB_TUNING_EXPERIMENT,
    MLFLOW_TUNING_EXPERIMENT,
    algorithm_display_name,
    build_pipeline_from_config,
    campaign_info,
    log_training_run_to_mlflow,
    prepare_split,
)
from ml.mlflow_tracking import configure_mlflow, log_educational_ml_suite_to_mlflow
from ml.scenario_simulation import (
    SIMULATABLE_FEATURES,
    build_intervention_narrative,
    predict_abandono_row,
)

__all__ = [
    "run_educational_ml_suite",
    "predict_taxa_abandono_em",
    "load_final_model_bundle",
    "configure_mlflow",
    "log_educational_ml_suite_to_mlflow",
    "EXPERIMENT_PLAN",
    "MLFLOW_TUNING_EXPERIMENT",
    "HGB_EXPERIMENT_GRID",
    "HGB_TUNING_EXPERIMENT",
    "algorithm_display_name",
    "build_pipeline_from_config",
    "campaign_info",
    "log_training_run_to_mlflow",
    "prepare_split",
    "SIMULATABLE_FEATURES",
    "build_intervention_narrative",
    "predict_abandono_row",
]
