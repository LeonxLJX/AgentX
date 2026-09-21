# AgentX — Nine-Module Full-Stack AI Outsourcing Toolkit

**One project. Every recurring high-tech freelance outsourcing type covered.**

AgentX is a modular full-stack project that packages the **recurring** high-tech
freelance outsourcing types (text classification, NLP phase detection,
probability calibration, job scheduling, path optimization, computational
geometry, data cleaning/OCR, financial modeling) into **9 independently
deliverable, demo-ready modules**, shipped with a REST API, a single-page demo
frontend and one-command Docker deployment.

> **For clients**: this is an "outsourcing ammunition kit" — whatever the job
> type, there is a matching module to demo, quote and deliver.
> **For developers**: every module wraps proven libraries (scikit-learn /
> SciPy / pandas / FastAPI) behind a clean, testable, persistable API.

---

## Module Overview

| # | Module | Core technology | Outsourcing fit | Reference price |
|---|--------|-----------------|-----------------|-----------------|
| 1 | **TextClassifier** | TF-IDF + LogisticRegression/SVM, multi-class EN/ZH | Comment / email / ticket classification | $300–600 |
| 2 | **PhaseDetect** | Sentiment lexicons + TF-IDF/KMeans topics + regex entities | Sentiment / topic / entity extraction | $300–800 |
| 3 | **Calibrator** | Platt Scaling / Isotonic + Brier/log-loss + reliability diagram | Prediction model calibration | $300–800 |
| 4 | **Scheduler** | Kahn topological sort + WSPT priorities + parallel machines | Scheduling / dispatch tools | $500–3000 |
| 5 | **Optimizer** | TSP (nearest-neighbour + 2-opt), CVRP (Clarke-Wright), knapsack DP | Delivery / route planning | $500–2000 |
| 6 | **Geometry** | Andrew convex hull, SAT collision, Pillow/NumPy raster ops | Hull / collision / image jobs | $200–800 |
| 7 | **ETLEngine** | pandas dedupe/fill/type inference/outliers + pytesseract OCR | Excel / OCR-to-data jobs | $50–300 |
| 8 | **FinanceModels** | Time-series features + factor library + linear/Ridge/AR(p) with backtest | Financial time-series jobs | $500–2000 |
| 9 | **REST API + Frontend** | FastAPI + vanilla JS SPA (SVG charts) + Docker | Full-stack delivery | $800–5000 |

## Tech Stack

```
Python 3.10+      language baseline
scikit-learn      text classification / calibration / KMeans topic clustering
NumPy / SciPy     numerics, linear algebra, TSP/VRP distance matrices
pandas            ETL cleaning and type inference
Pillow (+OpenCV)  image ops (gray / resize / blur / edge detection)
FastAPI+Uvicorn   REST API with auto Swagger docs
Pydantic          request/response validation
pydantic-settings typed configuration via environment variables
matplotlib        reliability diagrams and evaluation charts
pytesseract(opt)  OCR text recognition
Docker            one-command deployment
pytest            unit tests across all modules + API smoke tests
ruff              lint + format
mypy              static type checking
```

## Enterprise Features

Beyond the algorithm modules, AgentX ships with the production-grade
infrastructure an overseas engineering team would expect to maintain:

- **Structured logging** — every module logs through a shared
  `agentx.core.logging.get_logger` factory. The format is configurable via
  `AGENTX_LOG_FORMAT` (default `text`; switch to `json` for log shippers like
  Loki / CloudWatch / Datadog).
- **Typed exception hierarchy** — `agentx.core.exceptions` defines a base
  `AgentXError` plus per-module subclasses (`SchedulingError`,
  `OptimizationError`, `GeometryError`, `ETLError`, `FinanceError`,
  `CalibrationError`, `NLPError`, `TextClassificationError`,
  `ModelNotFittedError`, `PersistenceError`, `DependencyError`,
  `ValidationError`, `ConfigError`). The REST layer maps each one to a
  consistent JSON error envelope without inspecting message strings.
- **Configuration management** — `agentx.core.config.Settings` is a
  pydantic-settings model that reads every knob from environment variables
  (or a `.env` file). Feature flags include `AGENTX_ENABLE_METRICS`,
  `AGENTX_ENABLE_REQUEST_LOG`, `AGENTX_LOG_FORMAT`, `AGENTX_CORS_ORIGINS`
  and more.
- **Type safety** — public functions across all nine modules carry type
  annotations; the project is configured for `mypy --strict` with targeted
  relaxations for optional third-party stubs (`cv2`, `pytesseract`).
- **API observability** —
  - `GET /api/health` returns aggregate `ok`/`degraded` status plus per-module
    import probes (used by Docker healthcheck and load balancers).
  - `GET /api/metrics` returns per-route request counts, status code
    distribution and average latency (toggleable via
    `AGENTX_ENABLE_METRICS`).
  - `RequestLoggingMiddleware` logs every request with method, path, status
    and latency (toggleable via `AGENTX_ENABLE_REQUEST_LOG`).
- **Standardized error responses** — `ErrorResponse` (Pydantic) is the
  canonical envelope returned for any non-2xx response, with `error`,
  `detail` and `path` fields. Custom exception handlers translate
  `ValidationError` → 400, other `AgentXError` subclasses → 400, and any
  uncaught `Exception` → 500 (logged at `error` level with the full traceback).

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) dev & test dependencies
pip install -e ".[dev]"

# 3. Run the full test suite
pytest

# 4. Run any module example
python examples/demo_textclassifier.py
python examples/demo_finance.py
...

# 5. Start the REST API + frontend (http://localhost:8000)
uvicorn agentx.api.main:app --reload --port 8000
# Interactive docs: http://localhost:8000/docs
# Liveness probe:    http://localhost:8000/api/health
# Usage metrics:     http://localhost:8000/api/metrics

# 6. Docker one-command deployment
docker compose up -d
```

### Configuration

All runtime behavior is controlled via environment variables (or a `.env`
file at the repo root). A representative `.env`:

```
AGENTX_ENV=dev
AGENTX_LOG_LEVEL=INFO
AGENTX_LOG_FORMAT=text        # or "json" for log shippers
AGENTX_PORT=8000
AGENTX_CORS_ORIGINS=*
AGENTX_ENABLE_METRICS=true
AGENTX_ENABLE_REQUEST_LOG=true
```

See `agentx/core/config.py` for the full list of supported settings.

## Module Details

### 1. TextClassifier

```python
from agentx.textclassifier import TextClassifier
from agentx.textclassifier.dataset import load_ticket

texts, labels = load_ticket()          # built-in 4-intent support tickets
clf = TextClassifier().fit(texts, labels)
result = clf.predict("When will my package arrive?")[0]
print(result.label, result.confidence)  # logistics 0.97
clf.save("model.joblib")                # single-artifact delivery
```

Highlights: TF-IDF (sublinear, n-gram) + logistic regression; swappable
`LinearSVC`; Chinese handled via character n-grams (no segmentation dependency);
full probability distribution output; joblib persistence.
API: `POST /api/text/train`, `POST /api/text/classify`.

### 2. PhaseDetect

```python
from agentx.phasedetect import PhaseDetector
d = PhaseDetector(language="auto")
r = d.analyze("The service is extremely good! Mail a@b.com")
# sentiment=positive  score=2.1  entities={'EMAIL': ['a@b.com']}
d.fit_topics(corpus, n_topics=3)       # topic clustering
```

Highlights: bilingual (EN/ZH) sentiment lexicons with negation flips and
intensifier weighting; TF-IDF + KMeans topic clustering; regex entity
extraction (email / URL / phone / date / money); no heavy model downloads.
API: `POST /api/nlp/analyze`, `POST /api/nlp/topics`.

### 3. Calibrator

```python
from agentx.calibrator import Calibrator
cal = Calibrator(method="isotonic").fit(X, y)
proba = cal.predict_proba(X_test)                 # calibrated probabilities
gain = cal.calibration_gain(X_test, y_test)       # raw vs calibrated Brier/log-loss
```

Highlights: `CalibratedClassifierCV` (Platt / Isotonic) with k-fold
cross-fitting to avoid optimistic calibration; Brier score, log loss,
reliability bins and a reliability-diagram PNG deliverable.
API: `POST /api/calibrate`.

### 4. Scheduler

```python
from agentx.scheduler import JobScheduler
from agentx.core.schema import Job
jobs = [Job("A", 3, priority=5), Job("B", 2, dependencies=["A"])]
r = JobScheduler(machines=2).schedule(jobs)
print(r.makespan, r.machine_utilization)          # Gantt-ready timeline
```

Highlights: Kahn topological sort for DAG dependencies (cycle detection);
WSPT weighted-shortest-processing-time priority dispatch; earliest-free
parallel machine placement; Gantt-friendly output.
API: `POST /api/schedule`.

### 5. Optimizer

```python
from agentx.optimizer import TSP, CVRP, knapsack
route, dist = TSP(coords=coords).solve()          # nearest-neighbour + 2-opt
routes = CVRP(demands, capacity, coords).solve()  # Clarke-Wright savings
value, items = knapsack(weights, values, 10)      # 0/1 knapsack DP
```

Highlights: TSP construction + 2-opt improvement; capacity-constrained VRP
with per-route 2-opt polish; exact knapsack DP (adaptive float scaling) plus
greedy approximation; deterministic and JSON-friendly.
API: `POST /api/optimize/tsp|vrp|knapsack`.

### 6. Geometry

```python
from agentx.geometry import convex_hull, separation_vector, RasterImage
hull = convex_hull(points)                        # Andrew monotone chain
hit, mtv = separation_vector(poly_a, poly_b)      # SAT + minimum translation
img = RasterImage.from_file("a.png").grayscale().gaussian_blur(1.5).resize(400, 300)
edges = img.sobel_edges()                         # edge detection
```

Highlights: O(n log n) convex hull (pure Python); SAT collision detection with
MTV for collision *resolution*; AABB / circle / point-in-polygon tests;
chainable Pillow image pipeline with optional OpenCV acceleration.
API: `POST /api/geometry/hull|collision`.

### 7. ETLEngine

```python
from agentx.etl import DataCleaner, OcrEngine
cleaner = DataCleaner(df).clean()                 # dedupe/fill/types/outliers
print(cleaner.report())                           # audit report (deliverable)
engine = OcrEngine(lang="eng+chi_sim")            # OCR (optional dependency)
```

Highlights: pandas-driven cleaning — exact/subset dedupe, median/mode fill,
text normalization, bool/numeric/date type inference, IQR outlier detection and
capping, plus a cleaning audit report; OCR via pytesseract with graceful
degradation when missing.
API: `POST /api/etl/clean`.

### 8. FinanceModels

```python
from agentx.finance.models import ForecastModel
from agentx.finance.factors import rsi, sharpe_ratio, max_drawdown
m = ForecastModel(model="ar", lags=5).fit(prices, horizon=10)
pts = m.predict()                                 # 10-step forecast
print(m.backtest(prices, horizon=5))              # rolling backtest RMSE/MAE/MAPE
```

Highlights: vectorized SMA/EMA/volatility/z-score/trend regression; factor
library — momentum, mean reversion, RSI, Bollinger bands, Sharpe, max
drawdown; closed-form linear/Ridge/AR(p) forecasts with rolling-window
backtesting; fully auditable, no black-box optimizers.
API: `POST /api/finance/forecast|factors`.

### 9. REST API + Frontend

```bash
uvicorn agentx.api.main:app --port 8000
# Demo UI:      http://localhost:8000
# Swagger docs: http://localhost:8000/docs
# Docker:       docker compose up -d
```

Highlights: FastAPI with auto OpenAPI docs; open CORS for client integration;
all nine modules exposed as JSON endpoints; vanilla JS single-page frontend
(no CDN, works offline) with SVG visualizations (TSP route, convex hull,
Gantt chart, forecast curve, collision demo); multi-stage Docker build with
health check.

## Project Structure

```
agentx/
├── agentx/
│   ├── core/                 # shared infrastructure
│   │   ├── logging.py        # unified structured logging (text + JSON)
│   │   ├── exceptions.py     # typed exception hierarchy
│   │   ├── config.py         # pydantic-settings driven configuration
│   │   └── schema.py         # lightweight shared data models
│   ├── textclassifier/       # module 1
│   ├── phasedetect/          # module 2
│   ├── calibrator/           # module 3 (incl. reliability plotting)
│   ├── scheduler/            # module 4
│   ├── optimizer/            # module 5 (tsp / vrp / knapsack)
│   ├── geometry/             # module 6 (hull / collision / raster)
│   ├── etl/                  # module 7 (cleaner / ocr)
│   ├── finance/              # module 8 (timeseries / factors / models)
│   └── api/                  # module 9 (FastAPI + frontend)
│       ├── main.py           # app factory, health/metrics, middleware
│       └── routers/          # one router per module
├── examples/                 # one runnable example per module
├── tests/                    # unit tests + API smoke tests
├── tools/                    # static import checker (no deps)
├── docs/
│   ├── ARCHITECTURE.md       # architecture & design decisions
│   └── PITCH.md              # pitching scripts per module
├── Dockerfile / docker-compose.yml
├── requirements.txt / pyproject.toml
└── LICENSE (MIT)
```

## Development Priority (ROI order)

1. Core algorithms: TextClassifier → Calibrator → Optimizer → Scheduler → Geometry (fastest payoff)
2. Data layer: ETLEngine (incl. OCR)
3. Delivery layer: REST API → Frontend → Docker

## Pitching

Which outsourcing type each module maps to and how to present it to a client:
see **[docs/PITCH.md](docs/PITCH.md)**.

## License

[MIT](LICENSE) © AgentX Project Contributors
