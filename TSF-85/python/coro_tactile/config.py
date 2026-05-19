from pathlib import Path

# -----------------------------
# Sensor configuration
# -----------------------------
ATTR_NAME = "sensorCostume:label"
DEFAULT_VALUE = "Acceced"

ROWS = 18
COLS = 12
EXPECTED_SIZE = ROWS * COLS  # 216

CSV_NAME = "Nodes_id_filtered.csv"

# -----------------------------
# Paths
# -----------------------------
EXT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path.home() / "Documents"

SAVEDMODEL_DIR = EXT_ROOT / "data" / "CNN_Capacitive_Sensor"
MAXVAL_PATH = EXT_ROOT / "data" / "Maximum_value_network.npy"

# -----------------------------
# CSV naming
# -----------------------------
SUFFIX_DZ = "_deformations.csv"
SUFFIX_PRED = "_tactile_maps.csv"

# -----------------------------
# Heatmap settings
# -----------------------------
HEATMAP_MAX_FPS = 6.0
HEATMAP_MIN_DT = 1.0 / HEATMAP_MAX_FPS
HEATMAP_HOLD_LAST = True
