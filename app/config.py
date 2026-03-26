"""ChainInsight configuration."""

import os
from enum import Enum
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
CHARTS_DIR = DATA_DIR / "charts"

# Ensure directories exist
for d in [RAW_DIR, CLEAN_DIR, CHARTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'chaininsight.db'}")

# Authentication
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable must be set")

# CORS
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",") if o.strip()]

# Upload limits
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_SIZE = MAX_UPLOAD_MB * 1024 * 1024

# Rate limiting
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))


# --- Enums ---

class PipelineStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StockStatus(str, Enum):
    NORMAL = "Normal Stock"
    LOW = "Low Stock"
    OUT = "Out of Stock"


# --- Named constants ---

# Supply chain parameters
ORDERING_COST = 50       # $ per order
HOLDING_RATE = 0.25      # 25% of unit cost per year
MONTE_CARLO_SIMS = 5000
GA_POPULATION = 100
GA_GENERATIONS = 80
GA_MUTATION_RATE = 0.15
GA_CROSSOVER_RATE = 0.80

# Analysis constants
DSI_SENTINEL = 999
ABC_THRESHOLD_A = 80
ABC_THRESHOLD_B = 95
SUPPLY_RISK_WEIGHTS = (0.4, 0.4, 0.2)
SHAPIRO_SAMPLE_LIMIT = 500
RISK_LEVEL_BINS = [0, 0.3, 0.5, 0.7, 1.0]
RISK_LEVEL_LABELS = ["Low", "Medium", "High", "Critical"]

# Chart settings
CHART_DPI = 150
CHART_BG_COLOR = "#F0F2F6"
CHART_TEXT_COLOR = "#1B2A4A"
COLOR_PALETTE = {
    "primary": "#2E86C1",
    "success": "#27AE60",
    "danger": "#E74C3C",
    "warning": "#F39C12",
    "purple": "#8E44AD",
    "teal": "#1ABC9C",
    "gray": "#95A5A6",
}
STATUS_COLORS = {
    StockStatus.NORMAL.value: "#27AE60",
    StockStatus.LOW.value: "#F39C12",
    StockStatus.OUT.value: "#E74C3C",
}
