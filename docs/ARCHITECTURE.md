# AgentX Architecture

This document describes the layered architecture, module boundaries and key
design decisions of AgentX, for client review and team maintenance.

## 1. Overall Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client layer (optional)                  │
│   SPA demo frontend (HTML/CSS/JS + SVG)  |  Swagger /docs   │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP/JSON
┌──────────────────────────────▼──────────────────────────────┐
│                    REST API layer (FastAPI)                 │
│   /api/text  /api/nlp  /api/calibrate  /api/schedule        │
│   /api/optimize  /api/geometry  /api/etl  /api/finance      │
│   CORS · Pydantic validation · auto OpenAPI docs · health   │
└──────────────────────────────┬──────────────────────────────┘
                               │ in-process calls
┌──────────────────────────────▼──────────────────────────────┐
│                 Algorithm module layer (pure Python)        │
│  textclassifier  phasedetect  calibrator  scheduler         │
│  optimizer  geometry  etl  finance                          │
│  - every module importable, testable and persistable        │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    Infrastructure layer                      │
│  core (logging / data models) · NumPy/SciPy/pandas/sklearn  │
└─────────────────────────────────────────────────────────────┘
```

## 2. Layering Principles

- **Algorithm modules do not depend on the web stack**: the eight algorithm
  modules depend only on scientific libraries and `core`, so they work in any
  Python environment; the REST layer maps boundary types via Pydantic and
  `dataclasses`, keeping the algorithm modules free of FastAPI.
- **Shared data models** (`agentx/core/schema.py`): `ClassificationResult`,
  `NlpPhase`, `ScheduleEntry`, `Route`, `CleanReport`, `ForecastPoint` are
  reused across modules, keeping API response structures stable.
- **Observability**: all modules log through the unified `get_logger`; the API
  exposes `/api/health` for Docker health checks.

## 3. Module Design Notes

| Module | Key design decisions |
|--------|----------------------|
| TextClassifier | TF-IDF character n-grams for Chinese (no segmentation dependency); any sklearn estimator injectable; `decision_function` softmax normalization gives confidence for models without `predict_proba` |
| PhaseDetect | Sentiment is pure lexicon rules (interpretable, zero downloads); topics use TF-IDF + KMeans (needs a corpus); entities use multilingual regex; Chinese keywords via stop-character-filtered bigrams |
| Calibrator | Cross-fitted `CalibratedClassifierCV` gives *honest* out-of-fold probabilities; `calibration_gain` reports raw vs calibrated Brier/log-loss side by side, ready for a client report |
| Scheduler | Kahn topological sort + WSPT greedy (single-machine optimal); earliest-free machine placement; explicit errors for cycles, duplicate ids and unknown dependencies |
| Optimizer | TSP: nearest-neighbour construction + 2-opt; CVRP: Clarke-Wright savings + 2-opt; knapsack: exact DP (adaptive float-to-int scaling) + greedy; deterministic output |
| Geometry | Convex hull via Andrew's monotone chain (pure Python, deterministic); SAT collision returns a minimum translation vector (collision *resolution*, not just detection); raster ops are chainable, falling back to NumPy convolution when OpenCV is absent |
| ETLEngine | Every pipeline step is individually callable and `clean()` runs them in one shot; `CleanReport` records dedupe/fill/cap/type-fix counts as an audit deliverable; OCR is optional and degrades gracefully |
| FinanceModels | All time-series functions are vectorized and NaN-aware; factor library is separated from forecasting; forecasts use closed-form least squares (linear/Ridge/AR(p)) - auditable, no black-box optimizers; backtest is rolling-window |

## 4. Data Flow Example (Text Classification)

```
Client POST /api/text/classify
   -> Pydantic validation (ClassifyRequest)
   -> TextClassifier.predict(texts)
       -> Pipeline.predict -> TfidfVectorizer.transform -> LogisticRegression.predict
   -> ClassificationResult(dataclass) -> dict
   -> JSON response {results: [{text, label, confidence, scores}]}
```

## 5. Deployment & Operations

- **Docker**: multi-stage build (builder installs deps -> slim runtime);
  `HEALTHCHECK` pings `/api/health`; `docker compose up -d` starts everything.
- **Configuration**: `AGENTX_PORT` env var controls the port (default 8000);
  `AGENTX_ENV` is reserved for production/dev markers.
- **Testing**: `pytest` covers all eight algorithm modules plus API smoke tests
  (`TestClient`); all test data is seeded and reproducible.

## 6. Extension Guide

- New algorithm module: create a package under `agentx/` -> reuse
  `core/schema.py` and `core/logging.py` -> add a router in `api/routers/` ->
  register it in `main.py` -> add `examples/` and `tests/`.
- New model: inject any sklearn estimator into TextClassifier; inject any
  binary classifier into Calibrator.
- Localization: PhaseDetect lexicons and stop-word tables live at the top of
  `phasedetect/detector.py`; append per-language blocks.
