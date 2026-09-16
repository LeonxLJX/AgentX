# ============================================================================
# AgentX - a nine-module full-stack toolkit for high-tech freelance outsourcing
#
#  1. TextClassifier  - text classification with TF-IDF + linear models
#  2. PhaseDetect     - NLP phase detection (sentiment / topic / entity)
#  3. Calibrator      - probability calibration & model evaluation
#  4. Scheduler       - job scheduling (priorities, DAG dependencies, resources)
#  5. Optimizer       - path / delivery / knapsack optimization
#  6. Geometry        - computational geometry (convex hull, collision, raster)
#  7. ETLEngine       - data cleaning, dedup, type inference, OCR
#  8. FinanceModels   - financial time-series features, factors & forecasting
#  9. REST API + Web  - FastAPI backend, single-page demo frontend, Docker
# ============================================================================

"""AgentX: an all-in-one outsourcing ammunition kit.

Every module exposes a small, well-documented, dependency-friendly API so that
a specific freelance job (text classification, calibration, scheduling, path
planning, geometry, data cleaning or finance) can be fulfilled by dropping the
corresponding module into the delivery.
"""

__version__ = "1.0.0"
__all__ = [
    "textclassifier",
    "phasedetect",
    "calibrator",
    "scheduler",
    "optimizer",
    "geometry",
    "etl",
    "finance",
    "api",
]
