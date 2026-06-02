#!/usr/bin/env python3
"""
Deformation evaluation for the GRIPPER sensor, RIGHT + LEFT pads.

For each pad (separate CSV):
  - REST = the s1_Rx / s1_Ry / s1_Rz columns (undeformed reference geometry,
    constant across frames). No separate base_file, no frame-0 s1_xyz.
  - Prediction = rest rotated + translated by each frame's pose (simple, no
    local referencing):  pred = P_rest @ Rf.T + Tf
  - Deformation axis = Y:  dY = actual_y - predicted_y
  - Heatmap uses the LAST valid frame (gate: frame must have all 1200 sim nodes,
    so we never miss one of the 400 we use). Falls back a frame if not full.
  - Same NODES_FILE / 25x16 (=400) grid for both pads.

Figure 1 : 3D scatter at the LAST frame.
            Top row    = RIGHT pad (3 views)
            Bottom row = LEFT  pad (3 views)
            blue = predicted (rest @ pose),  red = actual (measured)
Figure 2 : dY heatmaps (RIGHT and LEFT) with a FRAME slider (scrub all frames).
"""

import pandas as pd
import numpy as np
import json
import onnxruntime as ort
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# PATHS  -  edit these
# ============================================================
CSV_RIGHT  = "/home/berith/Documents/TSF_85_emergency/examples/data_generated/TactileData_s1_mesh_state.csv"
CSV_LEFT   = "/home/berith/Documents/TSF_85_emergency/examples/data_generated/TactileData_s2_mesh_state.csv"
NODES_FILE = "/media/berith/BerithLAB/sensor_emergency/CNN_new/Nodes_id_filtered_1812.csv"
# NODES_FILE = "/media/berith/BerithLAB/sensor_emergency/Get_top_layer/Nodes_id_filtered.csv"

ROWS = 18
COLS = 12
GRID_NODES = ROWS * COLS      # nodes we actually use (from NODES_FILE)
FULL_SIM_NODES = 432         # the sim dumps this many; gate on it being present
DEFORM_AXIS = 0               # 0=X, 1=Y, 2=Z   (X)

# CNN (ONNX) for tactile prediction
# MODEL_DIR = Path("/media/berith/BerithLAB/sensor_emergency/CNN_coro_tactile")
# ONNX_FILE = MODEL_DIR / "best.onnx"
# MAX_FILE  = MODEL_DIR / "CNN_max.npy"          # holds the_max for X/the_max normalization

MODEL_DIR = Path("/media/berith/BerithLAB/sensor_emergency/CNN_new")
ONNX_FILE = MODEL_DIR / "best.onnx"
MAX_FILE  = MODEL_DIR / "norm_params.json"     # JSON holding {"the_max": ...}

MAP_ROWS, MAP_COLS = 7, 4                       # CNN output -> 7x4 (28 values)

# ============================================================
# HELPERS
# ============================================================

def load_node_order(nodes_file):
    """Load node IDs preserving file order (dedup but keep first occurrence)."""
    df = pd.read_csv(nodes_file, sep=None, engine="python")
    col = "node_id" if "node_id" in df.columns else df.columns[0]
    seen, ordered = set(), []
    for nid in df[col].dropna().astype(int):
        if nid not in seen:
            ordered.append(nid)
            seen.add(nid)
    print(f"Loaded {len(ordered)} node IDs from {nodes_file}")
    return ordered


def detect_xyz_cols(df):
    for x, y, z in [("s1_x", "s1_y", "s1_z"), ("x", "y", "z")]:
        if all(c in df.columns for c in [x, y, z]):
            return x, y, z
    raise ValueError(f"Cannot find XYZ columns. Available: {df.columns.tolist()}")


def detect_node_col(df):
    for c in ["node_id", "node"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find node_id column. Available: {df.columns.tolist()}")


def detect_frame_col(df):
    for c in ["frame", "Frame"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find frame column. Available: {df.columns.tolist()}")


def quat_to_R(w, x, y, z):
    """Quaternion (w,x,y,z) -> 3x3 rotation matrix. Normalizes first."""
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n == 0:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def pose_from_row(row):
    """Return (R, T) from a row's s1_Trans_* and s1_Ori_* fields."""
    T = np.array([row["s1_Trans_x"], row["s1_Trans_y"], row["s1_Trans_z"]], dtype=float)
    R = quat_to_R(row["s1_Ori_w"], row["s1_Ori_x"], row["s1_Ori_y"], row["s1_Ori_z"])
    return R, T


def scale_to_cube(pts, center, span, skip_axis=None):
    """Normalize points to a unit cube about `center` using per-axis `span`.
    A zero (flat) axis is left unscaled to avoid divide-by-zero.
    `skip_axis` (e.g. the deformation/pad-normal axis) is left in raw units,
    so deformation along it stays in physical metres instead of being
    distorted by the (near-zero) span of the flat normal axis."""
    out = np.zeros_like(pts)
    for k in range(3):
        if k == skip_axis or span[k] <= 1e-12:
            out[:, k] = pts[:, k]          # raw, no scaling
        else:
            out[:, k] = (pts[:, k] - center[k]) / span[k] + center[k]
    return out


def cube_params(pts):
    """Per-axis span (max-min) and bbox center for `pts`."""
    span   = [pts[:, k].max() - pts[:, k].min() for k in range(3)]
    center = [(pts[:, k].min() + pts[:, k].max()) / 2 for k in range(3)]
    return center, span


# ---- CNN (ONNX) helpers ----

def load_onnx_session():
    if not ONNX_FILE.exists():
        raise FileNotFoundError(f"ONNX model not found at {ONNX_FILE}")
    sess = ort.InferenceSession(str(ONNX_FILE),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print(f"[OK] ONNX providers: {sess.get_providers()}")
    return sess


def load_the_max():
    with open(MAX_FILE) as f:
        the_max = float(json.load(f)["the_max"])
    print(f"[INFO] the_max = {the_max}  (from {MAX_FILE.name})")
    return the_max


def onnx_predict(session, X_norm, batch_size=64):
    in_name  = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    exp = session.get_inputs()[0].shape
    if len(exp) == 4 and exp[1] == 1:
        feed = np.transpose(X_norm, (0, 3, 1, 2)).astype(np.float32)
    else:
        feed = X_norm.astype(np.float32)
    outs = []
    for s in range(0, feed.shape[0], batch_size):
        e = min(s + batch_size, feed.shape[0])
        outs.append(session.run([out_name], {in_name: feed[s:e]})[0])
    return np.concatenate(outs, axis=0)


def predict_all_frames(session, the_max, pad):
    """Run the CNN on every frame's dY grid -> pred_by_frame{f:(7,4)} + pred_absmax.
    NaNs in the dY map are zeroed before the net."""
    frames = pad["all_frames"]
    X = np.stack([np.nan_to_num(pad["grid_by_frame"][fr], nan=0.0) for fr in frames], axis=0)
    X = np.expand_dims(X, axis=-1).astype(np.float32)       # (N,ROWS,COLS,1)
    X_norm = (X / the_max).astype(np.float32)
    y_pred = onnx_predict(session, X_norm)                  # (N,28)
    pred_by_frame, pmax = {}, 0.0
    for i, fr in enumerate(frames):
        pm = np.asarray(y_pred[i]).reshape(MAP_ROWS, MAP_COLS)
        pred_by_frame[fr] = pm
        pmax = max(pmax, float(np.max(np.abs(pm))))
    pad["pred_by_frame_cnn"] = pred_by_frame
    pad["pred_absmax"] = pmax
    return pad


def frame_positions(df, fc, nc, xc, yc, zc, frame, node_order):
    """Per-node world XYZ for a single frame, reindexed to node_order.
    Returns (DataFrame indexed by node_id with cols x,y,z, pose_row)."""
    sub = df[df[fc] == frame]
    pose_row = sub.iloc[0]  # pose is constant within a frame
    pos = (sub[[nc, xc, yc, zc]]
           .rename(columns={nc: "node_id", xc: "x", yc: "y", zc: "z"})
           .set_index("node_id"))
    pos = pos.reindex(node_order)
    return pos, pose_row


def process_pad(csv_path, node_order, label):
    """
    Load one pad CSV and compute everything needed for plotting.
    Returns a dict with:
      label, all_frames, last_frame, valid (bool mask over node_order),
      common_ids, pred_by_frame{f:(M,3)}, act_by_frame{f:(M,3)},
      dY_last (M,), dY_grid (ROWS,COLS)
    """
    print(f"\n=== Processing {label} pad: {csv_path}")
    df = pd.read_csv(csv_path, sep=None, engine="python")
    fc = detect_frame_col(df)
    nc = detect_node_col(df)
    xc, yc, zc = detect_xyz_cols(df)

    df[fc] = pd.to_numeric(df[fc], errors="coerce")
    df[nc] = pd.to_numeric(df[nc], errors="coerce")
    df = df.dropna(subset=[fc, nc]).copy()
    df[fc] = df[fc].astype(int)
    df[nc] = df[nc].astype(int)

    all_frames = sorted(df[fc].unique())
    rest_frame = all_frames[0]
    print(f"  Frames {all_frames[0]}..{all_frames[-1]} ({len(all_frames)})  rest={rest_frame}")

    # Last valid frame (gate on full 1200 nodes)
    def full_node_count(frame):
        return int(df[df[fc] == frame][nc].nunique())

    last_frame = all_frames[-1]
    while last_frame > rest_frame and full_node_count(last_frame) < FULL_SIM_NODES:
        print(f"  [INFO] Frame {last_frame} has < {FULL_SIM_NODES} nodes; stepping back.")
        last_frame -= 1
    print(f"  Last frame used: {last_frame}")

    # REST points come from s1_Rx / s1_Ry / s1_Rz (undeformed reference geometry),
    # constant across frames; read at the last frame.
    for c in ["s1_Rx", "s1_Ry", "s1_Rz"]:
        if c not in df.columns:
            raise ValueError(f"{label}: missing {c}. Have: {df.columns.tolist()}")
    last_sub = df[df[fc] == last_frame]
    rest_R = (last_sub[[nc, "s1_Rx", "s1_Ry", "s1_Rz"]]
              .rename(columns={nc: "node_id", "s1_Rx": "x", "s1_Ry": "y", "s1_Rz": "z"})
              .set_index("node_id")
              .reindex(node_order))
    P_rest = rest_R[["x", "y", "z"]].values

    # Validity mask: nodes present at rest AND at the last frame
    last_pos, _ = frame_positions(df, fc, nc, xc, yc, zc, last_frame, node_order)
    P_last = last_pos[["x", "y", "z"]].values
    valid = ~np.isnan(P_rest).any(axis=1) & ~np.isnan(P_last).any(axis=1)
    common_ids = np.array(node_order)[valid]
    print(f"  Common nodes (file order): {valid.sum()}")

    # Precompute predicted + actual for EVERY frame (filtered to valid nodes).
    # Rotation + translation by each frame's pose, NO scaling.
    #     pred = P_rest @ Rf.T + Tf        (rotated rest cloud)
    #     act  = actual world XYZ for that frame
    # dY is then computed directly on the raw coordinates (metres).
    pred_by_frame, act_by_frame = {}, {}
    for fr in all_frames:
        pos_f, pose_f = frame_positions(df, fc, nc, xc, yc, zc, fr, node_order)
        Rf, Tf = pose_from_row(pose_f)
        pred_by_frame[fr] = (P_rest @ Rf.T + Tf)[valid]
        act_by_frame[fr]  = pos_f[["x", "y", "z"]].values[valid]

    # Per-frame dY grids (same sign convention as the last-frame block below),
    # plus the max |dY| over all frames for a fixed color scale on the slider.
    def dY_grid_for(fr):
        aa = act_by_frame[fr][:, DEFORM_AXIS]
        pp = pred_by_frame[fr][:, DEFORM_AXIS]
        dv = (pp - aa) if label == "RIGHT" else (pp - aa)
        if len(dv) == GRID_NODES:
            g = dv.reshape((ROWS, COLS))
        else:
            padded = np.full(GRID_NODES, np.nan)
            padded[:min(len(dv), GRID_NODES)] = dv[:GRID_NODES]
            g = padded.reshape((ROWS, COLS))
        # RIGHT pad: mirror columns (col 0 <-> col 11, ...) before feeding the net
        if label == "RIGHT":
            g = np.fliplr(g)
        return g

    grid_by_frame = {fr: dY_grid_for(fr) for fr in all_frames}
    absmax_all = 0.0
    for g in grid_by_frame.values():
        m = np.nanmax(np.abs(g))
        if np.isfinite(m):
            absmax_all = max(absmax_all, m)

    # dY at the last frame.
    # Convention (consistent with dY_grid_for above):
    #   RIGHT = predicted - actual ;  LEFT = actual - predicted.
    a = act_by_frame[last_frame][:, DEFORM_AXIS]
    p = pred_by_frame[last_frame][:, DEFORM_AXIS]
    if label == "RIGHT":
        dvec = p - a
    else:
        dvec = a - p
    print(f"  dY(last)  min={dvec.min():.6f}  max={dvec.max():.6f}  mean={dvec.mean():.6f}")
    if len(dvec) != GRID_NODES:
        print(f"  [WARN] Expected {GRID_NODES} nodes, got {len(dvec)}. Heatmap will pad/trim.")
        padded = np.full(GRID_NODES, np.nan)
        padded[:len(dvec)] = dvec
        dY_grid = padded.reshape((ROWS, COLS))
    else:
        dY_grid = dvec.reshape((ROWS, COLS))

    # RIGHT pad: mirror columns (col 0 <-> col 11, ...) to match dY_grid_for
    if label == "RIGHT":
        dY_grid = np.fliplr(dY_grid)

    return dict(label=label, all_frames=all_frames, last_frame=last_frame,
                valid=valid, common_ids=common_ids,
                pred_by_frame=pred_by_frame, act_by_frame=act_by_frame,
                dY_last=dvec, dY_grid=dY_grid,
                grid_by_frame=grid_by_frame, absmax_all=absmax_all)


# ============================================================
# LOAD + PROCESS BOTH PADS
# ============================================================
node_order = load_node_order(NODES_FILE)
pads = [
    process_pad(CSV_RIGHT, node_order, "RIGHT"),
    process_pad(CSV_LEFT,  node_order, "LEFT"),
]

# CNN prediction for every frame of both pads
session = load_onnx_session()
the_max = load_the_max()
for pad in pads:
    predict_all_frames(session, the_max, pad)
# Shared prediction color scale across both pads + all frames
pred_absmax = max(p["pred_absmax"] for p in pads) or 1e-6

# Per-pad fixed axis limits (over all frames, so the box is comfortable)
def pad_limits(pad):
    allp = np.concatenate(list(pad["pred_by_frame"].values()) +
                          list(pad["act_by_frame"].values()), axis=0)
    allp = allp[~np.isnan(allp).any(axis=1)]
    return ((allp[:, 0].min(), allp[:, 0].max()),
            (allp[:, 1].min(), allp[:, 1].max()),
            (allp[:, 2].min(), allp[:, 2].max()))

# ============================================================
# FIGURE 1 - two rows (RIGHT top, LEFT bottom), LAST frame only
#   blue = predicted (rest @ last-frame pose),  red = actual (measured)
# ============================================================
views = [(30, 45), (90, 0), (0, 90)]
labels_v = ["3D View", "Top (XZ)", "Side (YZ)"]

fig1 = plt.figure(figsize=(18, 11))
for r, pad in enumerate(pads):                       # row 0 = RIGHT, row 1 = LEFT
    xlim, ylim, zlim = pad_limits(pad)
    lf = pad["last_frame"]
    pb = pad["pred_by_frame"][lf]
    ab = pad["act_by_frame"][lf]
    for c, (elev, azim) in enumerate(views):
        idx = r * 3 + c + 1
        ax = fig1.add_subplot(2, 3, idx, projection="3d")
        ax.scatter(pb[:, 0], pb[:, 1], pb[:, 2],
                   c="blue", s=18, alpha=0.7, label="Predicted (rest @ pose)")
        ax.scatter(ab[:, 0], ab[:, 1], ab[:, 2],
                   c="red", s=18, alpha=0.7, label="Actual (measured)")
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        ax.set_xlabel("X", fontsize=8); ax.set_ylabel("Y", fontsize=8); ax.set_zlabel("Z", fontsize=8)
        ax.set_title(f"{pad['label']} (frame {lf}) - {labels_v[c]}", fontsize=10)
        ax.view_init(elev=elev, azim=azim)
        if c == 0:
            ax.legend(fontsize=8)

fig1.suptitle("Predicted (blue) vs Actual (red)  |  last frame", fontsize=13)
fig1.subplots_adjust(top=0.92, hspace=0.25)

# ============================================================
# FIGURE 2 - 2x2 with a FRAME slider
#   top row    = dY heatmaps          (RIGHT | LEFT)
#   bottom row = CNN tactile prediction (RIGHT | LEFT, 7x4)
#   Figure 1 stays static at the last frame; this one scrubs all frames.
# ============================================================
shared_frames = sorted(set(pads[0]["all_frames"]) & set(pads[1]["all_frames"]))
f_min, f_max = shared_frames[0], shared_frames[-1]
init_frame = f_max  # start on the last frame, matching Figure 1

fig2, ax2 = plt.subplots(2, 2, figsize=(12, 11))
art = {}
for col, pad in enumerate(pads):                 # col 0 = RIGHT, col 1 = LEFT
    am = pad["absmax_all"] or 1e-6
    # top: dY heatmap
    im_t = ax2[0, col].imshow(pad["grid_by_frame"][init_frame], cmap="RdBu_r",
                              aspect="auto", vmin=-am, vmax=am)
    cbar_t = fig2.colorbar(im_t, ax=ax2[0, col], fraction=0.046, pad=0.04)
    conv = "Predicted - Actual" if pad["label"] == "RIGHT" else "Actual - Predicted"
    cbar_t.set_label(f"dY ({conv})", fontsize=10)
    ax2[0, col].set_xlabel(f"Col (0-{COLS-1})  ->  Rx ascending", fontsize=9)
    ax2[0, col].set_ylabel(f"Row (0-{ROWS-1})  ->  Rz descending", fontsize=9)
    ax2[0, col].set_xticks(range(COLS)); ax2[0, col].set_yticks(range(ROWS))
    # bottom: CNN prediction (7x4)
    im_b = ax2[1, col].imshow(pad["pred_by_frame_cnn"][init_frame], cmap="jet",
                              aspect="equal", origin="upper",
                              vmin=-20, vmax=pred_absmax)
    fig2.colorbar(im_b, ax=ax2[1, col], fraction=0.046, pad=0.04, label="tactile")
    ax2[1, col].set_title(f"{pad['label']} - prediction (7x4)", fontsize=10)
    ax2[1, col].set_xticks(range(MAP_COLS)); ax2[1, col].set_yticks(range(MAP_ROWS))
    txt = [[ax2[1, col].text(c, r, "", ha="center", va="center", color="white", fontsize=7)
            for c in range(MAP_COLS)] for r in range(MAP_ROWS)]
    art[col] = dict(pad=pad, im_t=im_t, im_b=im_b, txt=txt, ax_t=ax2[0, col])

sup2 = fig2.suptitle("", fontsize=12)


def _update_fig2(frame):
    fr = int(round(frame))
    for col in (0, 1):
        a = art[col]; pad = a["pad"]
        g = pad["grid_by_frame"].get(fr)
        pm = pad["pred_by_frame_cnn"].get(fr)
        if g is not None:
            a["im_t"].set_data(g)
            a["ax_t"].set_title(f"{pad['label']}  dY ({ROWS}x{COLS})  frame {fr}\n"
                                f"min={np.nanmin(g):.5f}  max={np.nanmax(g):.5f}  "
                                f"mean={np.nanmean(g):.5f}", fontsize=9)
        if pm is not None:
            a["im_b"].set_data(pm)
            for r in range(MAP_ROWS):
                for c in range(MAP_COLS):
                    a["txt"][r][c].set_text(f"{pm[r, c]:.1f}")
    sup2.set_text(f"dY (top) -> CNN tactile prediction (bottom)   |   Frame {fr} / {f_max}")
    fig2.canvas.draw_idle()


fig2.subplots_adjust(bottom=0.09, top=0.90, hspace=0.30, wspace=0.30)
s2_ax = fig2.add_axes([0.25, 0.03, 0.5, 0.025])
slider2 = Slider(ax=s2_ax, label="Frame", valmin=f_min, valmax=f_max,
                 valinit=init_frame, valstep=1)
slider2.on_changed(_update_fig2)
_update_fig2(init_frame)

plt.show()
print("\nDone.")
