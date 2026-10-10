from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.schemas import BacktestRequest, DatasetProvenance, SimulationResult


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backtest: BacktestRequest = Field(
        default_factory=lambda: BacktestRequest(start=date(2022, 1, 3), end=date(2025, 1, 1))
    )
    algorithm: Literal["xgboost", "q_learning"] = "xgboost"
    horizon: int = Field(1, ge=1, le=5)
    evaluation_anchor_horizon: int | None = Field(None, ge=1, le=5)
    holdout_fraction: float = Field(0.2, ge=0.15, le=0.3)
    folds: int = Field(3, ge=2, le=5)
    n_estimators: int = Field(160, ge=20, le=600)
    max_depth: int = Field(3, ge=1, le=6)
    learning_rate: float = Field(0.05, ge=0.01, le=0.3)
    min_edge_bps: float = Field(5, ge=0, le=200, allow_inf_nan=False)
    probability_threshold: float = Field(0.55, ge=0.5, le=0.9)
    feature_set: Literal["basic", "technical"] = "basic"
    entry_rule: Literal["return_and_probability", "probability_only"] = "return_and_probability"
    subsample: float = Field(1, ge=0.5, le=1, allow_inf_nan=False)
    colsample_bytree: float = Field(1, ge=0.5, le=1, allow_inf_nan=False)
    min_child_weight: float = Field(5, ge=1, le=20, allow_inf_nan=False)
    reg_lambda: float = Field(5, ge=0, le=50, allow_inf_nan=False)
    training_window_months: int = Field(0, ge=0, le=60)
    recency_half_life_days: float = Field(0, ge=0, le=730, allow_inf_nan=False)
    validation_scheme: Literal["blocks", "monthly"] = "blocks"
    seed: int = Field(42, ge=0, le=2147483647)
    episodes: int = Field(20, ge=2, le=60)
    drawdown_penalty: float = Field(0, ge=0, le=2, allow_inf_nan=False)
    refresh_data: bool = False
    parent_id: str | None = Field(None, pattern=r"^[0-9a-f]{24}$")

    @model_validator(mode="after")
    def daily(self):
        if self.evaluation_anchor_horizon is not None and self.evaluation_anchor_horizon < self.horizon:
            raise ValueError("Evaluation anchor horizon must cover the forecast horizon.")
        if self.backtest.timeframe != "1d":
            raise ValueError("AI research currently uses daily bars. Select 1d.")
        if self.algorithm == "q_learning" and self.horizon != 1:
            raise ValueError("Q-learning uses one-bar reward transitions; select horizon 1.")
        if 0 < self.training_window_months < 3:
            raise ValueError("Training windows must be at least 3 months, or 0 for all earlier history.")
        if self.algorithm == "q_learning" and (
            self.recency_half_life_days or self.training_window_months or self.validation_scheme != "blocks"
        ):
            raise ValueError("Recency weighting and monthly rolling training currently support XGBoost.")
        return self


class PredictionMetrics(BaseModel):
    samples: int
    mae: float | None = None
    rmse: float | None = None
    directional_accuracy: float | None = None
    rank_correlation: float | None = None
    brier_score: float | None = None
    log_loss: float | None = None
    calibration_error: float | None = None
    reliability: list[dict] = Field(default_factory=list)


class Evaluation(BaseModel):
    prediction: PredictionMetrics
    prediction_baselines: dict[str, PredictionMetrics]
    strategy: SimulationResult
    baselines: dict[str, SimulationResult]
    regimes: list[dict]
    predictions: list[dict]
    activity: dict | None = None


class RetrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["latest", "same_dates"] = "latest"


class ResearchResult(BaseModel):
    id: str
    created_at: str
    config: ResearchRequest
    provenance: DatasetProvenance
    code_version: str
    feature_version: str
    libraries: dict[str, str]
    features: list[str]
    training: dict
    walk_forward: list[dict]
    validation: Evaluation
    holdout: Evaluation
    importance: list[dict]
    drift: dict
    promotion: dict
    latest_prediction: dict
    warnings: list[str]
    assumptions: list[str]
    search: dict | None = None
    retraining: dict | None = None


class ResearchJob(BaseModel):
    id: str
    kind: Literal["train", "search"] = "train"
    search_id: str | None = None
    status: Literal["queued", "running", "completed", "failed", "interrupted"]
    progress: str
    created_at: str
    result_id: str | None = None
    error: str | None = None


class GridRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    minimum: float = Field(allow_inf_nan=False)
    maximum: float = Field(allow_inf_nan=False)
    step: float = Field(gt=0, allow_inf_nan=False)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    research: ResearchRequest
    trials: int = Field(12, ge=2, le=48)
    objective: Literal["sharpe", "net_return"] = "sharpe"
    mode: Literal["random", "coarse_to_fine", "local", "grid"] = "random"
    grid: dict[str, GridRange] = Field(default_factory=dict)

    @model_validator(mode="after")
    def supervised(self):
        if self.research.algorithm != "xgboost":
            raise ValueError("Automatic parameter search currently supports XGBoost. Select XGBoost.")
        if self.mode == "grid":
            from app.ml.grid import grid_space

            grid_space(self)
        elif self.grid:
            raise ValueError("Range controls apply to exhaustive grid search only.")
        return self


class SearchResult(BaseModel):
    id: str
    status: Literal["running", "completed", "failed"]
    created_at: str
    config: SearchRequest
    space: dict[str, list[int | float | str]]
    selection_policy: str
    refinement: dict | None = None
    total_combinations: int | None = None
    code_version: str
    provenance: DatasetProvenance
    validation_start: str
    validation_end: str
    holdout_start: str
    winner_id: str | None
    selected_trial: int | None
    trials: list[dict]
    error: str | None
    warnings: list[str]
