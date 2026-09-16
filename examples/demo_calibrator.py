# ============================================================================
# Example: Calibrator - probability calibration with honest metrics
# Run:  python examples/demo_calibrator.py
# ============================================================================

"""Trains a deliberately miscalibrated model, recalibrates it, and prints the
Brier / log-loss gain plus a reliability-diagram PNG."""

import numpy as np
from sklearn.datasets import make_classification

from agentx.calibrator import Calibrator
from agentx.calibrator.metrics import plot_reliability

if __name__ == "__main__":
    # Synthetic binary problem (seeded for reproducibility).
    X, y = make_classification(
        n_samples=1500, n_features=8, n_informative=5,
        n_redundant=1, flip_y=0.15, random_state=42,
    )
    split = int(len(X) * 0.7)
    X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

    for method in ("sigmoid", "isotonic"):
        cal = Calibrator(method=method, random_state=42).fit(X_tr, y_tr)
        gain = cal.calibration_gain(X_te, y_te)
        print(f"\n=== method={method} ===")
        print(f"raw        : Brier={gain['raw']['brier']:.4f}  "
              f"log_loss={gain['raw']['log_loss']:.4f}")
        print(f"calibrated : Brier={gain['calibrated']['brier']:.4f}  "
              f"log_loss={gain['calibrated']['log_loss']:.4f}")
        for row in gain["reliability"]:
            print(f"  bin={row['bin_center']:.2f}  predicted={row['mean_predicted']:.3f}  "
                  f"observed={row['fraction_positive']:.3f}  n={row['count']}")

    # Reliability diagram artifact (deliverable for client reports).
    cal = Calibrator(method="isotonic", random_state=42).fit(X_tr, y_tr)
    proba = cal.predict_proba(X_te)[:, 1]
    path = plot_reliability(y_te, proba, "outputs/reliability.png")
    print(f"\nreliability diagram saved to {path}")
