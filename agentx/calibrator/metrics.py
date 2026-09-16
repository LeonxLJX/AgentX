# ============================================================================
# AgentX.calibrator.metrics - proper scoring rules & reliability analysis
# ============================================================================

"""Evaluation metrics for calibrated probabilities.

- :func:`brier_score`      - mean squared error of probabilities vs outcomes
- :func:`log_loss_score`   - cross-entropy (proper scoring rule)
- :func:`reliability_curve`- binned mean prediction vs observed frequency
- :func:`plot_reliability` - matplotlib reliability diagram (PNG artifact)
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import log_loss


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Brier score (lower is better): mean of (p - y)^2.

    Parameters
    ----------
    y_true : array-like
        Binary labels (0/1).
    y_proba : array-like
        Predicted probability of the positive class.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_proba, dtype=float).ravel()
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_proba must have the same length")
    return float(np.mean((yp - yt) ** 2))


def log_loss_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Log loss / cross-entropy (lower is better)."""
    yt = np.asarray(y_true).ravel()
    yp = np.clip(np.asarray(y_proba, dtype=float).ravel(), 1e-12, 1 - 1e-12)
    stacked = np.column_stack([1.0 - yp, yp])
    return float(log_loss(yt, stacked))


def reliability_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    """Bin probabilities and compare mean prediction vs observed frequency.

    Returns
    -------
    list[dict]
        One entry per bin: ``bin_center``, ``mean_predicted``,
        ``fraction_positive`` and ``count``. Bins with no samples are dropped.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_proba, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    rows: List[Dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (yp >= lo) & (yp < hi)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin_center": float((lo + hi) / 2),
                "mean_predicted": float(yp[mask].mean()),
                "fraction_positive": float(yt[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return rows


def plot_reliability(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: str,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
) -> str:
    """Render a reliability diagram to a PNG file.

    The perfect-calibration line (y = x) is drawn for reference; points below
    the line are over-confident, points above are under-confident.

    Returns
    -------
    str
        Path of the written PNG file.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless-safe backend
    import matplotlib.pyplot as plt

    curve = reliability_curve(y_true, y_proba, n_bins=n_bins)
    if not curve:
        raise ValueError("no populated bins - cannot plot reliability")

    centers = [r["bin_center"] for r in curve]
    predicted = [r["mean_predicted"] for r in curve]
    observed = [r["fraction_positive"] for r in curve]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(centers, observed, "o-", color="#2563eb", label="Observed frequency")
    ax.plot(centers, predicted, "s-", color="#f59e0b", label="Mean prediction", alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed fraction of positives")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
