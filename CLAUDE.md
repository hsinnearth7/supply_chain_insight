# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChainInsight — End-to-end supply chain analytics platform combining:
1. **Hierarchical Demand Forecasting** — Nixtla-format, 6-model routing ensemble (MAPE 10.3%), 4-layer hierarchy with MinTrace reconciliation
2. **Curriculum-Learning RL** — Multi-product inventory optimization (PPO+SAC) with 3-phase curriculum
3. **MLOps Infrastructure** — Feature store (AP > CP), Evidently drift monitoring, Pandera data contracts
4. **Live Mode** — FastAPI + React SPA + SQLite with real-time WebSocket updates, 8-step ETL, 28+ charts

## Running

### Quick Start (Docker)

```bash
docker compose up --build
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000/docs
```

### Development

```bash
# Install with all dependencies (PEP 621)
pip install -e ".[dev]"

# Set up environment
cp .env.example .env

# React Frontend (dev mode)
cd frontend && npm install && npm run dev

# FastAPI Backend
uvicorn app.main:app --reload --port 8000
```

### Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v                    # 163 tests across 14 files
pytest tests/ -v -m "not slow"      # Skip slow integration tests
ruff check app/ tests/
mypy app/
```

### Generate Data

```bash
python -m app.forecasting.data_generator                 # Generate + validate
python -m app.forecasting.data_generator --validate-only # Validate only
```

## Configuration

### YAML Config (`configs/chaininsight.yaml`)

Primary configuration for all modules. Loaded via `app/settings.py` with `@lru_cache`:

```python
from app.settings import get_data_config, get_model_config, get_rl_config
```

Key sections: `data`, `model`, `evaluation`, `rl`, `supply_chain`, `monitoring`, `chart`, `server`

### Environment Variables (`.env`)

Runtime config: `API_KEY`, `CORS_ORIGINS`, `DATABASE_URL`, `MAX_UPLOAD_MB`, `RATE_LIMIT_PER_MINUTE`

## Architecture

```
Synthetic Data Generator (Nixtla format, M5 properties, Pandera validation)
         │
         ├── 4-Layer Hierarchy: National(1) → Warehouse(3) → Category(60) → SKU(200)
         │
    ┌────┴────────────────────────────────────────────┐
    ▼                                                  ▼
Forecasting Pipeline                           RL Pipeline
    │                                                  │
    ├── Feature Store (offline/online)         ├── Multi-Product Env (Gymnasium)
    ├── 6 Models (routing ensemble)            ├── Curriculum Learning (3 phases)
    ├── Hierarchical Reconciliation (MinTrace) ├── Classical Baselines (Newsvendor, (s,S), EOQ)
    ├── Walk-Forward CV (12-fold)              └── PPO + SAC (stable-baselines3)
    ├── Conformal Prediction Intervals
    └── Wilcoxon + Cohen's d significance
         │
    ┌────┴────┐
    ▼         ▼
MLOps      Live Mode
    │         │
    ├── Drift Monitor (KS, PSI, MAPE)    ├── FastAPI + React SPA
    └── Auto-retrain triggers             ├── WebSocket real-time updates
                                          ├── 8-step ETL pipeline
                                          └── 28+ charts
```

### Key Modules (v2.0 Additions)

| Module | Path | Description |
|--------|------|-------------|
| **Settings** | `app/settings.py` | YAML config loader with `@lru_cache`, section accessors |
| **Logging** | `app/logging.py` | structlog setup (`setup_logging()`, `get_logger()`) |
| **Seed** | `app/seed.py` | Global seed management (random, numpy, torch, CUDA) |
| **Data Generator** | `app/forecasting/data_generator.py` | Nixtla-format synthetic data, 5 M5 properties, 4-layer hierarchy |
| **Data Contracts** | `app/forecasting/contracts.py` | Pandera schemas for Y_df, S_df, X_future, X_past, forecasts |
| **Forecast Models** | `app/forecasting/models.py` | 6 models + ForecastModelFactory (Strategy pattern) |
| **Hierarchy** | `app/forecasting/hierarchy.py` | Aggregation + MinTrace reconciliation |
| **Evaluation** | `app/forecasting/evaluation.py` | Walk-forward CV, Wilcoxon, Cohen's d, conformal prediction |
| **Feature Store** | `app/forecasting/feature_store.py` | Offline/online dual-mode, eventual consistency |
| **Drift Monitor** | `app/forecasting/drift_monitor.py` | KS data drift, PSI prediction drift, MAPE concept drift |
| **Multi-Product Env** | `app/rl/multi_product_env.py` | Gymnasium env: continuous action space, stochastic lead time |
| **Curriculum** | `app/rl/curriculum.py` | 3-phase progressive training (1→3→5 products) |
| **RL Baselines** | `app/rl/baselines.py` | Newsvendor (stockpyl), (s,S), EOQ policy evaluation |

### Key Modules (Original)

| Module | Path | Description |
|--------|------|-------------|
| Config | `app/config.py` | Settings, enums (`PipelineStatus`, `StockStatus`), constants |
| Auth | `app/auth.py` | API Key authentication (`require_api_key` dependency) |
| Enrichment | `app/pipeline/enrichment.py` | Shared `enrich_base()` (DSI, Coverage, Demand Intensity) |
| ETL | `app/pipeline/etl.py` | `ETLPipeline` — 8-step cleaning with schema validation |
| Stats | `app/pipeline/stats.py` | `StatisticalAnalyzer` — charts 0-8, KPIs |
| Supply Chain | `app/pipeline/supply_chain.py` | `SupplyChainAnalyzer` — charts 9-14 |
| ML Engine | `app/pipeline/ml_engine.py` | `MLAnalyzer` — charts 15-22, no data leakage |
| Orchestrator | `app/pipeline/orchestrator.py` | `PipelineOrchestrator` — coordinates all stages |
| RL Environment | `app/rl/environment.py` | Gymnasium `InventoryEnv` — 5-state, 5-action |
| RL Agents | `app/rl/agents/*.py` | Q-Learning, SARSA, DQN, PPO, A2C, GA-RL Hybrid |
| RL Trainer | `app/rl/trainer.py` | `RLTrainer` — seed control, convergence detection |
| RL Evaluator | `app/rl/evaluator.py` | `RLEvaluator` — charts 23-28, KPI comparison |
| API Entry | `app/main.py` | FastAPI app with CORS, WS routes, watchdog, SPA mount |
| DB Models | `app/db/models.py` | SQLAlchemy with FK relationships |
| React Frontend | `frontend/` | Vite + React 18 + TypeScript + Tailwind + Recharts |

## Forecasting Pipeline

### Data Format (Nixtla Long Format)

- `Y_df`: `(unique_id, ds, y)` — demand time series
- `S_df`: `(unique_id, warehouse, category, subcategory)` — hierarchy metadata
- `X_future`: `(unique_id, ds, price, promo, day_of_week, month)` — future-known features
- `X_past`: `(unique_id, ds, is_stockout)` — historical-only features

### 6 Forecasting Models

| Model | Class | Use Case |
|-------|-------|----------|
| Naive MA-30 | `NaiveMovingAverage` | Baseline reference |
| SARIMAX | `SARIMAXForecaster` | Seasonal/intermittent demand |
| XGBoost | `XGBoostForecaster` | Feature interactions |
| LightGBM | `LightGBMForecaster` | Best single model (MAPE 12.1%) |
| Chronos-2 ZS | `ChronosForecaster` | Cold-start, zero-shot |
| Routing Ensemble | `RoutingEnsemble` | Routes by history length + intermittency |

### Routing Logic

- History < 60 days → Chronos-2 ZS (cold-start)
- Intermittent demand (>30% zeros) → SARIMAX
- Mature SKU → LightGBM

### Evaluation Protocol

- 12-fold walk-forward CV (monthly retrain, 14-day horizon)
- Wilcoxon signed-rank test (α=0.05) vs Naive baseline
- Cohen's d effect size (S<0.5, M=0.5-0.8, L>0.8)
- Conformal prediction intervals (90% target coverage)

## RL Pipeline

### Curriculum Learning

| Phase | Products | Demand | Lead Time | Steps |
|-------|----------|--------|-----------|-------|
| 1 | 1 | Normal | Fixed | 50K |
| 2 | 3 | Seasonal | Fixed | 100K |
| 3 | 5 | Intermittent | Stochastic | 200K |

### Classical Baselines

- **Newsvendor** (stockpyl): theoretical optimal cost
- **(s,S) Policy**: reorder-point with fixed order quantity
- **EOQ Policy**: economic order quantity

## MLOps

### Feature Store

- Offline: batch materialization for training
- Online: single-row retrieval for serving
- AP > CP design (eventual consistency, documented in ADR-001)

### Drift Monitoring

- **Data drift**: KS-test (threshold 0.05)
- **Prediction drift**: PSI (threshold 0.1)
- **Concept drift**: MAPE trend (retrain if >20% for 7 consecutive days)

## Documentation

| Document | Path | Content |
|----------|------|---------|
| Model Card | `docs/model_card.md` | Mitchell et al. FAT* 2019 format |
| Reproducibility | `docs/reproducibility.md` | NeurIPS 2019 ML reproducibility standard |
| Failure Modes | `docs/failure_modes.md` | 5 components, 5-level degradation |
| ADR-001 | `docs/adr/001-cap-tradeoff-feature-store.md` | AP > CP for feature store |
| ADR-002 | `docs/adr/002-routing-ensemble-over-stacking.md` | Routing vs stacking/blending |
| ADR-003 | `docs/adr/003-multi-warehouse-degradation.md` | Graceful degradation strategy |

## Testing

163 tests across 14 files:

| File | Tests | Coverage |
|------|-------|----------|
| `test_data_generator.py` | 27 | Data generation, schemas, M5 properties |
| `test_forecasting_models.py` | 14 | All 6 models, factory pattern |
| `test_evaluation.py` | 21 | Metrics, CV, statistical tests |
| `test_hierarchy.py` | 5 | Aggregation, reconciliation |
| `test_feature_store.py` | 11 | Offline/online modes |
| `test_drift_monitor.py` | 8 | 3 drift types |
| `test_property_based.py` | 7 | Hypothesis invariants |
| `test_multi_product_env.py` | 14 | Gymnasium env |
| `test_rl_baselines.py` | 12 | Classical policies |
| `test_config.py` | 16 | YAML config loading |
| Original test files | 28 | ETL, API, RL, pipeline |

## Development Guidelines
- Always JSON-serialize datetime objects and numpy types before sending through WebSocket. Use `SafeEncoder` (see root CLAUDE.md).
- Test WebSocket connections immediately after implementation to catch routing/serialization issues.
- After editing Python files, verify syntax with `python -m py_compile <file>`.
- Run `pytest tests/ -v` after code changes, not as a final step.
- For the React frontend: run `npm run build` to catch TypeScript errors early.

## Security

- **Authentication**: API Key via `X-API-Key` header (all endpoints except `/api/health`)
- **CORS**: Configurable origins via `CORS_ORIGINS` env var (not `*`)
- **Path Traversal**: `_safe_path()` validates all file paths stay within base directory
- **Upload Security**: Size limits, filename sanitization, CSV validation
- **Rate Limiting**: In-memory per-IP rate limiter

## Dependencies

Managed via `pyproject.toml` (PEP 621):
- Core: `pandas`, `numpy`, `scipy`, `scikit-learn`, `pandera`, `pyyaml`, `structlog`
- Forecasting: `statsforecast`, `hierarchicalforecast`, `lightgbm`, `xgboost`
- RL: `gymnasium`, `torch`, `stable-baselines3`, `stockpyl`
- MLOps: `evidently`
- Live: `fastapi`, `uvicorn`, `sqlalchemy`, `websockets`, `watchdog`
- Frontend: `react`, `react-router-dom`, `recharts`, `zustand`, `tailwindcss`, `vite`
- Dev: `pytest`, `hypothesis`, `ruff`, `mypy`, `pre-commit`
- Optional: `chronos-t5` (Chronos-2 foundation model, ~300MB)
