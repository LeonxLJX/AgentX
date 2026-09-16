# ============================================================================
# AgentX.calibrator.calibrator - Calibrator implementation
# ============================================================================

"""Post-hoc probability calibration for binary classifiers.

Implementation
-------------
The calibrator wraps scikit-learn's ``CalibratedClassifierCV``:

- ``method="sigmoid"`` : Platt scaling - fits a logistic function on the
  scores. Best when the miscalibration has a known sigmoid shape.
- ``method="isotonic"`` : isotonic regression - non-parametric, more flexible,
  requires more calibration data.

The base model is trained and calibrated on disjoint folds (``cv``), which
avoids optimistic calibration. Probabilities are therefore honest out-of-fold
estimates suitable for risk decisions and reporting.

The module also exposes a convenience ``evaluate`` returning Brier score,
log loss and a reliability table so a client report can be produced directly.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from agentx.calibrator.metrics import brier_score, log_loss_score, reliability_curve
from agentx.core import get_logger

logger = get_logger("agentx.calibrator")


class Calibrator:
    """Recalibrate any binary classifier's probability output.

    Parameters
    ----------
    base_model : estimator, optional
        Any scikit-learn classifier exposing ``fit`` and
        ``decision_function``/``predict_proba``. Defaults to logistic
        regression on the raw features.
    method : {"sigmoid", "isotonic"}
        Calibration method (see class docstring).
    cv : int or "prefit"
        Number of calibration folds, or ``"prefit"`` to calibrate an already
        fitted ``base_model`` without retraining.
    random_state : int
        Seed for reproducible fold splits.

    Attributes
    ----------
    calibrated_ : CalibratedClassifierCV
        The fitted calibration wrapper.
    """

    def __init__(
        self,
        base_model: Optional[Any] = None,
        method: str = "isotonic",
        cv: Any = 5,
        random_state: int = 42,
    ) -> None:
        if method not in {"sigmoid", "isotonic"}:
            raise ValueError("method must be 'sigmoid' or 'isotonic'")
        self.base_model = base_model or LogisticRegression(max_iter=2000)
        self.method = method
        self.cv = cv
        self.random_state = random_state
        self.calibrated_: Optional[CalibratedClassifierCV] = None
        self.classes_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ fit
    def fit(self, X: Any, y: Any) -> "Calibrator":
        """Train and calibrate the model on features ``X`` and labels ``y``.

        ``y`` must be binary (0/1). Multi-class calibration is intentionally
        out of scope; for multi-class problems calibrate per class (one-vs-rest).
        """
        y_arr = np.asarray(y)
        if y_arr.ndim != 1 or set(np.unique(y_arr)) - {0, 1}:
            raise ValueError("Calibrator supports binary labels {0, 1} only")

        self.calibrated_ = CalibratedClassifierCV(
            estimator=self.base_model,
            method=self.method,
            cv=self.cv,
            n_jobs=-1,
        )
        self.calibrated_.fit(X, y_arr)
        self.classes_ = self.calibrated_.classes_
        logger.info(
            "calibration fitted (method=%s, cv=%s, base=%s)",
            self.method,
            self.cv,
            type(self.base_model).__name__,
        )
        return self

    # ------------------------------------------------------------- predict
    def predict_proba(self, X: Any) -> np.ndarray:
        """Return calibrated probabilities of shape ``(n_samples, 2)``."""
        return self._require_fitted().predict_proba(X)

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        """Hard predictions using a decision threshold on the positive class."""
        proba = self.predict_proba(X)
        positive = proba[:, -1]  # positive class is the last column
        return (positive >= threshold).astype(int)

    def calibration_gain(self, X: Any, y: Any) -> Dict[str, Any]:
        """Compare raw vs calibrated probability quality.

        Both quantities are measured on the same ``X``/``y``:

        - **raw**     : out-of-fold predictions of a freshly fitted base model
          via ``cross_val_predict`` (honest, no in-sample optimism).
        - **calibrated** : predictions of the fitted calibrator.

        Returns
        -------
        dict
            Raw and calibrated Brier score / log loss plus the reliability
            table (``bin_center``, ``mean_predicted``, ``fraction_positive``).
        """
        cal = self._require_fitted()
        y_arr = np.asarray(y)

        # Honest raw baseline: out-of-fold predictions of the base model.
        raw_model = type(self.base_model)(**self._init_params_safe())
        cv = self.cv if isinstance(self.cv, int) else 5
        method = "predict_proba" if hasattr(raw_model, "predict_proba") else "decision_function"
        raw_pred = cross_val_predict(raw_model, X, y_arr, cv=cv, method=method, n_jobs=-1)
        if method == "decision_function":
            if raw_pred.ndim == 1:
                raw_pred = np.column_stack([-raw_pred, raw_pred])
            exp = np.exp(raw_pred - raw_pred.max(axis=1, keepdims=True))
            raw_pred = exp / exp.sum(axis=1, keepdims=True)
        raw_proba_pos = raw_pred[:, -1] if raw_pred.ndim > 1 else raw_pred

        cal_proba = cal.predict_proba(X)[:, -1]

        return {
            "raw": {
                "brier": brier_score(y_arr, raw_proba_pos),
                "log_loss": log_loss_score(y_arr, raw_proba_pos),
            },
            "calibrated": {
                "brier": brier_score(y_arr, cal_proba),
                "log_loss": log_loss_score(y_arr, cal_proba),
            },
            "reliability": reliability_curve(y_arr, cal_proba),
        }

    def evaluate(self, X: Any, y: Any) -> Dict[str, Any]:
        """Evaluate calibrated probabilities on held-out data."""
        cal = self._require_fitted()
        y_arr = np.asarray(y)
        proba = cal.predict_proba(X)[:, -1]
        return {
            "brier": brier_score(y_arr, proba),
            "log_loss": log_loss_score(y_arr, proba),
            "reliability": reliability_curve(y_arr, proba),
            "n_samples": int(len(y_arr)),
        }

    # --------------------------------------------------------- persistence
    def save(self, path: str) -> str:
        """Persist the fitted calibrator as a joblib artifact."""
        self._require_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({"calibrated": self.calibrated_, "classes": self.classes_}, path)
        logger.info("calibrator saved to %s", path)
        return path

    @classmethod
    def load(cls, path: str) -> "Calibrator":
        """Load a calibrator saved with :meth:`save`."""
        payload = joblib.load(path)
        obj = cls()
        obj.calibrated_ = payload["calibrated"]
        obj.classes_ = payload["classes"]
        logger.info("calibrator loaded from %s", path)
        return obj

    # ------------------------------------------------------------- helpers
    def _require_fitted(self) -> CalibratedClassifierCV:
        if self.calibrated_ is None:
            raise RuntimeError("calibrator is not fitted - call .fit(X, y) first")
        return self.calibrated_

    def _init_params_safe(self) -> Dict[str, Any]:
        """Copy constructor kwargs of the base model for a fresh instance."""
        init = getattr(self.base_model, "get_params", lambda: {})()
        return {k: v for k, v in init.items() if k != "n_jobs"}

    @staticmethod
    def _proba_of(model: Any, X: Any) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(X))
        raw = np.asarray(model.decision_function(X))
        if raw.ndim == 1:
            return np.column_stack([-raw, raw])
        return raw
