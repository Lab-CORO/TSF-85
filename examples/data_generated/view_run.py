"""
Simple frame-by-frame viewer for one TSF-85 run.

Standalone: by default it looks for the CSVs in the SAME folder as this
script. Drop view_run.py into the folder that contains the
TactileData_*_deformations.csv / _tactile_maps.csv files and run it.

To view the other sensor, change SENSOR to "s2". To point somewhere else,
set DATA_DIR to an absolute path.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# =====================================================================
# EDIT THESE (optional)
# =====================================================================
SENSOR = "s1"          # "s1" (RIGHT) or "s2" (LEFT)

# Folder holding the CSVs. Defaults to this script's own directory so the
# viewer is fully standalone. Override with an absolute path if needed.
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

DZ_CSV   = os.path.join(DATA_DIR, f"TactileData_{SENSOR}_deformations.csv")
PRED_CSV = os.path.join(DATA_DIR, f"TactileData_{SENSOR}_tactile_maps.csv")

# Frame ranges to display. Each tuple is (start_frame, end_frame), inclusive.
# Leave RANGES = [] to show ALL frames in the CSV.
RANGES = [
]
# =====================================================================

ROWS, COLS = 18, 12
PRED_ROWS, PRED_COLS = 7, 4

for _p in (DZ_CSV, PRED_CSV):
    if not os.path.exists(_p):
        raise SystemExit(
            f"CSV not found:\n  {_p}\n"
            f"Put this script in the same folder as the TactileData_{SENSOR}_*.csv "
            f"files, or set DATA_DIR / SENSOR at the top."
        )

# Load both CSVs. First column = time_sec, second = frame, rest = values.
dz_raw   = np.genfromtxt(DZ_CSV,   delimiter=",", skip_header=1)
pred_raw = np.genfromtxt(PRED_CSV, delimiter=",", skip_header=1)

times_all  = dz_raw[:, 0]
frames_all = dz_raw[:, 1].astype(int)
dz_all     = dz_raw[:, 2:]
pred_all   = pred_raw[:, 2:]

n_all = min(len(dz_all), len(pred_all))
dz_all, pred_all = dz_all[:n_all], pred_all[:n_all]
times_all, frames_all = times_all[:n_all], frames_all[:n_all]

# Empty RANGES = show everything. Otherwise filter to rows whose frame
# falls inside any of the requested windows.
if not RANGES:
    mask = np.ones(n_all, dtype=bool)
else:
    mask = np.zeros(n_all, dtype=bool)
    for lo, hi in RANGES:
        mask |= (frames_all >= lo) & (frames_all <= hi)

dz     = dz_all[mask]
pred   = pred_all[mask]
times  = times_all[mask]
frames = frames_all[mask]
n      = len(dz)

if n == 0:
    raise SystemExit(
        f"No frames matched. RANGES={RANGES}, "
        f"CSV covers frames {int(frames_all.min())}..{int(frames_all.max())}."
    )

if RANGES:
    print(f"Loaded {n_all} frames total; showing {n} inside {RANGES}.")
else:
    print(f"Loaded {n_all} frames total; showing all.")
print(f"  Frame range on disk: {int(frames_all.min())}..{int(frames_all.max())}")

# Fixed color scales across the visible subset.
dz_vmin,   dz_vmax   = float(dz.min()),   float(dz.max())
pred_vmin, pred_vmax = float(pred.min()), float(pred.max())

fig, (ax_dz, ax_pred) = plt.subplots(1, 2, figsize=(11, 7))
plt.subplots_adjust(bottom=0.18, wspace=0.3)

im_dz = ax_dz.imshow(dz[0].reshape(ROWS, COLS), cmap="jet",
                     vmin=dz_vmin, vmax=dz_vmax, aspect="auto")
ax_dz.set_title(f"Deformation  ({ROWS}x{COLS})")
fig.colorbar(im_dz, ax=ax_dz, fraction=0.046, pad=0.04)

im_pred = ax_pred.imshow(pred[0].reshape(PRED_ROWS, PRED_COLS), cmap="jet",
                         vmin=pred_vmin, vmax=pred_vmax, aspect="auto")
ax_pred.set_title(f"Predicted tactile  ({PRED_ROWS}x{PRED_COLS})")
fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

footer = fig.text(0.5, 0.10,
                  f"frame={frames[0]}   t={times[0]:.3f}s   (1/{n})",
                  ha="center", fontsize=10)

slider = Slider(ax=fig.add_axes([0.15, 0.04, 0.7, 0.03]),
                label="frame", valmin=0, valmax=n - 1, valinit=0, valstep=1)

def on_slide(val):
    i = int(val)
    im_dz.set_data(dz[i].reshape(ROWS, COLS))
    im_pred.set_data(pred[i].reshape(PRED_ROWS, PRED_COLS))
    footer.set_text(f"frame={frames[i]}   t={times[i]:.3f}s   ({i + 1}/{n})")
    fig.canvas.draw_idle()

slider.on_changed(on_slide)
plt.show()
