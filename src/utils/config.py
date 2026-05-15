import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar .env desde la raíz del proyecto
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# DATA_ROOT viene del .env, si no existe usa "data" como fallback
DATA_ROOT = Path(os.getenv("DATA_ROOT", "data")).resolve()

# Subcarpetas fijas del proyecto
RAW_DIR = DATA_ROOT / "dataraw"
PROCESSED_DIR = DATA_ROOT / "processed"
GRAPHS_DIR = DATA_ROOT / "graphs"

# Archivo de failures (siempre dentro de RAW_DIR)
FAILURES_FILE = RAW_DIR / "failures_07_25.csv"

# Parámetros globales del pipeline
ENTITY_COL = "CERT"
PERIOD_COL = "period"