"""
Per-sensor pipeline state and operations.

The TSF-85 extension supports 1 or 2 sensors simultaneously (e.g. the two
pads of a gripper). All state that varies per-sensor lives in a
SensorChannel. The Extension object holds 1 or 2 of these and dispatches
the physics step / inference / writing to each.

Things that are SHARED across sensors (so they live in the Extension, not
in SensorChannel):
  - the ONNX session (one model, both sensors call it)
  - the sensor_config.json data (node_ids, the_max)
  - settings, panel, heatmap UI (heatmap currently shows channel 0 only)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class SensorChannel:
    """All per-sensor state needed to compute dY, run inference, and write
    the three CSVs for one sensor.

    Attribute groups:

    - identity:
        tag        : "" for single-sensor mode, "s1"/"s2" for two-sensor mode.
                     Used as a filename infix so two-sensor mode produces
                     <base>_s1_*.csv and <base>_s2_*.csv side by side.
        root_path  : path of the sensor case prim in USD.

    - cached USD handles (filled by Extension._ensure_deformable_cached):
        mesh_path, pos_attr, rest_attr, vel_attr, rest_attr_name

    - latest computed values (updated each physics step):
        last_dz           : flat list of EXPECTED_SIZE floats
        last_pred         : flat numpy array of PRED_SIZE floats
        last_pred_frame   : the frame index that pred was computed from
        last_time         : timeline time at the latest dz compute
        last_frame        : frame index at the latest dz compute
        last_processed_frame : guards against running the step twice on
                               the same frame

    - CSV writers (paired per output type; opened in _start_logging,
      closed in _close_files_now):
        dz_path / dz_fh / dz_writer
        pred_path / pred_fh / pred_writer
        mesh_path_csv / mesh_fh / mesh_writer

    - async inference queue (per channel so each sensor's pipeline is
      independent of the other):
        infer_queue : deque of (t, frame_idx, grid_25x16)
        pending_close : if Stop happens while items are still in the queue,
                        flip this true so the inference loop will close
                        the files once the queue drains.
    """
    tag: str = ""
    root_path: Optional[str] = None

    # USD handles (set by the cache logic)
    mesh_path: Optional[str] = None
    pos_attr: Any = None
    rest_attr: Any = None
    vel_attr: Any = None
    rest_attr_name: Optional[str] = None

    # latest values
    last_dz: Any = None
    last_pred: Any = None
    last_pred_frame: Optional[int] = None
    last_time: float = 0.0
    last_frame: int = 0
    last_processed_frame: Optional[int] = None

    # CSV writers
    dz_path: Optional[Path] = None
    dz_fh: Any = None
    dz_writer: Any = None
    pred_path: Optional[Path] = None
    pred_fh: Any = None
    pred_writer: Any = None
    mesh_path_csv: Optional[Path] = None
    mesh_fh: Any = None
    mesh_writer: Any = None

    # async inference
    infer_queue: deque = field(default_factory=lambda: deque(maxlen=8))
    pending_close: bool = False

    # ---- small helpers ----
    def clear_cache(self):
        """Drop cached USD handles so they get re-resolved on the next
        physics step. Called when the sensor selection changes."""
        self.mesh_path = None
        self.pos_attr = None
        self.rest_attr = None
        self.vel_attr = None
        self.rest_attr_name = None

    def clear_writers(self):
        """Null out writer handles after close (file objects themselves
        are closed by the caller)."""
        self.dz_path = None
        self.dz_fh = None
        self.dz_writer = None
        self.pred_path = None
        self.pred_fh = None
        self.pred_writer = None
        self.mesh_path_csv = None
        self.mesh_fh = None
        self.mesh_writer = None
