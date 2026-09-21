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
│  core (logging / exceptions / config / data models)         │
│  · NumPy/SciPy/pandas/sklearn                                │
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
- **Typed exceptions** (`agentx/core/exceptions.py`): every module raises a
  specific `AgentXError` subclass so callers (and the REST layer) can catch
  failures with the right granularity without inspecting message strings.
- **Configuration** (`agentx/core/config.py`): a pydantic-settings `Settings`
  singleton drives feature flags, log format, CORS, metrics and more from
  environment variables.
- **Observability**: all modules log through the unified `get_logger`
  (text or JSON format); the API exposes `/api/health` for Docker health
  checks and `/api/metrics` for usage telemetry; `RequestLoggingMiddleware`
  logs every request.

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
- **Configuration**: `agentx/core/config.py` defines a pydantic-settings
  `Settings` model loaded from environment variables (or a `.env` file). All
  knobs are prefixed `AGENTX_*` — supported settings include:
  - `AGENTX_ENV` (`dev` | `staging` | `prod`)
  - `AGENTX_LOG_LEVEL` / `AGENTX_LOG_FORMAT` (`text` | `json`)
  - `AGENTX_HOST` / `AGENTX_PORT` / `AGENTX_CORS_ORIGINS`
  - `AGENTX_ENABLE_METRICS` / `AGENTX_ENABLE_REQUEST_LOG`
  - `AGENTX_MODEL_CACHE_SIZE`
- **Logging**: every module logs through `agentx.core.get_logger`, which
  configures the root `agentx` logger once. Records can be emitted as plain
  text (default) or as single-line JSON for log shippers (`AGENTX_LOG_FORMAT=json`).
- **Observability**:
  - `GET /api/health` returns `{"status": "ok"|"degraded", "modules": {...}}`
    with one import probe per algorithm module (used by Docker `HEALTHCHECK`).
  - `GET /api/metrics` returns per-route request counts, response code
    distribution and average latency (toggleable via `AGENTX_ENABLE_METRICS`).
  - `RequestLoggingMiddleware` emits one INFO line per request with method,
    path, status code and latency (toggleable via `AGENTX_ENABLE_REQUEST_LOG`).
- **Error handling**: `agentx.core.exceptions` defines a typed exception
  hierarchy rooted at `AgentXError`. The API registers handlers that map
  `ValidationError` → HTTP 400, other `AgentXError` subclasses → HTTP 400, and
  any uncaught `Exception` → HTTP 500 (logged at `error` level with the full
  traceback). All non-2xx responses share the `ErrorResponse` envelope
  (`{"error", "detail", "path"}`).
- **Testing**: `pytest` covers all eight algorithm modules plus API smoke tests
  (`TestClient`); all test data is seeded and reproducible.

## 6. Extension Guide

- New algorithm module: create a package under `agentx/` -> reuse
  `core/schema.py`, `core/logging.py` and `core/exceptions.py` (raise a
  module-specific `AgentXError` subclass) -> add a router in `api/routers/` ->
  register it in `main.py` (and append a probe tuple entry in
  `_MODULE_PROBES`) -> add `examples/` and `tests/`.
- New model: inject any sklearn estimator into TextClassifier; inject any
  binary classifier into Calibrator.
- Localization: PhaseDetect lexicons and stop-word tables live at the top of
  `phasedetect/detector.py`; append per-language blocks.
