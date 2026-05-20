from pathlib import Path

# -----------------------------
# Extension identity
# -----------------------------
# IMPORTANT: must match the [package] name in extension.toml.
EXT_NAME = "TSF_85_Ext"
SETTINGS_PREFIX = f"/exts/{EXT_NAME}"

# -----------------------------
# Sensor configuration
# -----------------------------
ATTR_NAME = "sensorCostume:label"
DEFAULT_VALUE = "Acceced"

# Input grid (deformable-mesh sample nodes).
# Row-major order: the first COLS node ids in Nodes_id_filtered.csv
# form row 0, the next COLS form row 1, and so on.
ROWS = 25
COLS = 16
EXPECTED_SIZE = ROWS * COLS  # 400

# CNN tactile output map is 7 x 4 = 28 values
PRED_ROWS = 7
PRED_COLS = 4
PRED_SIZE = PRED_ROWS * PRED_COLS  # 28

CSV_NAME = "Nodes_id_filtered.csv"

# -----------------------------
# Paths
# -----------------------------
# EXT_ROOT is resolved at runtime from the Kit extension manager
# (see extension.py -> _resolve_ext_root). This fallback is only used
# if that resolution fails for some reason.
#   config.py lives at: <EXT_ROOT>/python/TSF_85_Ext/config.py
#   so parents[2] climbs to <EXT_ROOT>.
EXT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path.home() / "Documents"


def data_dir(ext_root: Path) -> Path:
    return Path(ext_root) / "data"


def onnx_model_path(ext_root: Path) -> Path:
    return data_dir(ext_root) / "tactile_cnn.onnx"


def norm_params_path(ext_root: Path) -> Path:
    return data_dir(ext_root) / "norm_params.json"


def nodes_csv_path(ext_root: Path) -> Path:
    return data_dir(ext_root) / CSV_NAME


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
