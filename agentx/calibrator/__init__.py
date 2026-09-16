# ============================================================================
# AgentX.calibrator - Module 3: Probability Calibration
# ============================================================================

"""Probability calibration & model evaluation for prediction freelancing.

When a client has an existing model whose probabilities are miscalibrated
(over-confident or under-confident), this module re-calibrates it with Platt
scaling or isotonic regression and quantifies the gain with proper scoring
rules (Brier score, log loss) and a reliability diagram.

Typical outsourcing fit
-----------------------
- Calibrating credit-risk / churn / CTR models
- Fixing over-confident deep-learning softmax outputs
- Delivering calibration curves as part of model evaluation reports

Quick start
-----------
>>> from agentx.calibrator import Calibrator
>>> from agentx.calibrator.metrics import brier_score
>>> calibrator = Calibrator(method="isotonic").fit(X, y)
>>> proba = calibrator.predict_proba(X_test)
>>> brier_score(y_test, proba[:, 1])
0.0831
"""

from agentx.calibrator.calibrator import Calibrator
from agentx.calibrator.metrics import (
    brier_score,
    log_loss_score,
    plot_reliability,
    reliability_curve,
)

__all__ = ["Calibrator", "brier_score", "log_loss_score", "reliability_curve", "plot_reliability"]
