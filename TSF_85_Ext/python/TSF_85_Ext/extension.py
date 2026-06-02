import omni
import omni.ext
import asyncio
import csv
import json
from pathlib import Path
from collections import deque
import re

import numpy as np
import omni.ui as ui
import omni.usd
import omni.timeline
import omni.physx as _physx
import carb.settings
from pxr import Sdf, Usd, UsdGeom, PhysxSchema

import onnxruntime as ort

try:
    from PIL import Image, ImageDraw, ImageFont  # Pillow
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

from .config import *
from .ui_panel import CoRoPanel
from .sensor_pipeline import SensorChannel


class Extension(omni.ext.IExt):

    # =========================================================
    # STARTUP / SHUTDOWN
    # =========================================================
    def on_startup(self, ext_id):
        self._running = True
        self._ext_id = ext_id

        self._ctx = omni.usd.get_context()
        self._sel = self._ctx.get_selection()
        self._timeline = omni.timeline.get_timeline_interface()

        # Resolve the extension root robustly from the extension manager.
        self._ext_root = self._resolve_ext_root(ext_id)
        print("[TSF-85][INFO] Extension root:", self._ext_root)

        # Settings (script control)
        self._settings = carb.settings.get_settings()
        self._settings_prefix = SETTINGS_PREFIX  # /exts/TSF_85_Ext

        # Headless mode: when true, the panel-open gate is dropped so a
        # Python launcher script can drive the extension without ever
        # opening the UI. Set via carb settings:
        #   carb.settings.get_settings().set("/exts/TSF_85_Ext/headless", True)
        self._headless = bool(self._settings.get(f"{self._settings_prefix}/headless") or False)
        if self._headless:
            print("[TSF-85][INFO] Headless mode enabled. UI panel not required.")

        # Output settings defaults
        self._output_dir = DEFAULT_OUTPUT_DIR
        self._base_name = "TactileData"

        # Sensor selection defaults
        self._active_sensor_root_path = None
        self._awaiting_new_sensor_pick = True

        # Optional second sensor. Stays None unless the script (or, later,
        # the GUI) sets /exts/TSF_85_Ext/sensor_root_2 to a valid prim path.
        # When non-None, the extension records both sensors in parallel:
        # sensor 1 runs through the existing single-sensor code path, and
        # sensor 2 runs through the parallel channel code path. CSV files
        # gain _s1 / _s2 infixes when sensor 2 is active.
        self._sensor2 = None
        self._last_settings_sensor_root_2 = None

        # UI
        self._window = None
        self._panel = None

        # Heatmap UI (file-based)
        self._heatmap_img = None
        self._heatmap_png_a = (self._ext_root / "data" / "_pred_heatmap_a.png").resolve()
        self._heatmap_png_b = (self._ext_root / "data" / "_pred_heatmap_b.png").resolve()
        self._heatmap_png_toggle = False
        self._last_heatmap_update_t = -1e9
        # Wipe any leftover heatmap PNGs from a previous session so we never
        # show stale frames at startup. Belt-and-suspenders cleanup also runs
        # on timeline stop (see _cleanup_heatmap_pngs).
        self._cleanup_heatmap_pngs()

        # Node ids + normalization (unified loader; prefers sensor_config.json
        # and falls back to legacy Nodes_id_filtered.csv + norm_params.json)
        self._nodes_ids = []
        self._max_val = None
        self._load_sensor_data()

        # Deformable cache
        self._clear_deformable_cache()

        # ONNX model
        self._session = None
        self._infer_input_name = None
        self._load_onnx_model()

        # Latest values
        self._last_dz = None
        self._last_time = 0.0
        self._last_frame = 0
        self._last_pred = None
        self._last_pred_frame = None

        # Logging
        self._dz_path = None
        self._pred_path = None
        self._mesh_path = None
        self._dz_fh = None
        self._dz_writer = None
        self._pred_fh = None
        self._pred_writer = None
        self._mesh_fh = None
        self._mesh_writer = None
        self._logging_active = False
        self._flush_every = 50
        self._since_flush = 0

        # Record gate (script-driven via carb setting
        # /exts/TSF_85_Ext/record_active). When False, processing still runs
        # (mesh read, deformation, inference) so state stays warm, but NO rows
        # are written to any CSV. Default False: nothing is recorded until the
        # launcher sets record_active = True (e.g. just before closing the
        # gripper) and back to False (e.g. after the open finishes).
        self._record_active = False

        # Per-file enable flags. Toggled from the UI checkboxes.
        # Inference + heatmap run regardless of these; these only control
        # whether the corresponding CSV is written.
        self._log_dz_enabled = True
        self._log_pred_enabled = True
        self._log_mesh_enabled = True

        # Ensure one row per timeline frame
        self._last_processed_frame = None

        # Track play state to detect Stop events
        self._was_playing = False

        # Inference queue: (time_sec, frame_idx, grid25x16)
        self._infer_queue = deque(maxlen=8)
        self._infer_task = asyncio.ensure_future(self._inference_loop())

        # Defer-close mechanism so pred rows are written even if Stop happens quickly
        self._pending_close = False

        # Track last applied settings
        self._last_settings_sensor_root = None
        self._last_settings_out_dir = None
        self._last_settings_base_name = None

        # Apply settings once at startup
        self._apply_settings(initial=True)

        # PhysX subscription
        self._physx = _physx.get_physx_interface()
        self._physx_sub = None
        try:
            self._physx_sub = self._physx.subscribe_physics_step_events(self._on_physics_step)
            print("[TSF-85][INFO] Subscribed to PhysX physics-step events.")
        except Exception as e:
            print("[TSF-85][ERROR] Failed to subscribe to PhysX physics-step events:", e)

        # Poll loop for selection + UI updates + stop detection
        self._task = asyncio.ensure_future(self._poll_loop())

        # Window menu entry
        self._menu_item = None
        try:
            import omni.kit.menu.utils as menu_utils
            self._menu_item = menu_utils.MenuItemDescription(
                name="TSF-85 Tactile Sensor",
                onclick_fn=self.show_window,
            )
            menu_utils.add_menu_items([self._menu_item], "Window")
            print("[TSF-85][INFO] Window menu item added: Window -> TSF-85 Tactile Sensor")
        except Exception as e:
            print("[TSF-85][WARN] Failed to add Window menu item:", e)

        print("[TSF-85][INFO] Extension started (build v23c: rot+trans only, axis X, sign pred-act (both))")

    def on_shutdown(self):
        self._running = False

        try:
            if getattr(self, "_task", None):
                self._task.cancel()
        except Exception:
            pass
        try:
            if getattr(self, "_infer_task", None):
                self._infer_task.cancel()
        except Exception:
            pass

        self._physx_sub = None
        self._pending_close = False
        self._close_files_now()

        try:
            if getattr(self, "_menu_item", None) is not None:
                import omni.kit.menu.utils as menu_utils
                menu_utils.remove_menu_items([self._menu_item], "Window")
        except Exception:
            pass

        self._window = None
        self._session = None
        self._clear_deformable_cache()

        print("[TSF-85][INFO] Extension shutdown complete.")

    # =========================================================
    # EXTENSION ROOT RESOLUTION
    # =========================================================
    def _resolve_ext_root(self, ext_id) -> Path:
        """Ask the Kit extension manager for this extension's folder.
        Falls back to the path math in config.py if unavailable."""
        try:
            import omni.kit.app
            mgr = omni.kit.app.get_app().get_extension_manager()
            p = mgr.get_extension_path(ext_id)
            if p:
                return Path(p).resolve()
        except Exception as e:
            print("[TSF-85][WARN] Could not resolve ext path from manager:", e)
        return Path(EXT_ROOT)

    # =========================================================
    # DEBUG PRINT ONCE
    # =========================================================
    def _dbg_once(self, key: str, msg: str):
        if not hasattr(self, "_dbg_seen"):
            self._dbg_seen = set()
        if key not in self._dbg_seen:
            self._dbg_seen.add(key)
            print(msg)

    # =========================================================
    # SETTINGS APPLY (script <-> extension bridge)
    # =========================================================
    def _apply_settings(self, initial: bool = False):
        s = getattr(self, "_settings", None)
        if s is None:
            return
        prefix = self._settings_prefix

        sensor_root = s.get(f"{prefix}/sensor_root")
        sensor_root = str(sensor_root) if sensor_root else None
        if sensor_root and sensor_root != self._last_settings_sensor_root:
            self._last_settings_sensor_root = sensor_root
            self._active_sensor_root_path = sensor_root
            self._awaiting_new_sensor_pick = False
            self._clear_deformable_cache()

        # Optional sensor 2. If set to a non-empty string, build a channel;
        # if cleared (set to empty / None), drop it. Idempotent: only acts
        # when the value actually changes.
        sensor_root_2 = s.get(f"{prefix}/sensor_root_2")
        sensor_root_2 = str(sensor_root_2) if sensor_root_2 else None
        if sensor_root_2 != self._last_settings_sensor_root_2:
            self._last_settings_sensor_root_2 = sensor_root_2
            if sensor_root_2:
                # New / changed sensor 2 -- (re)create the channel. Tag "s2"
                # so its CSVs get the right infix. Sensor 1's tag gets set
                # below in _compute_output_paths logic.
                self._sensor2 = SensorChannel(tag="s2", root_path=sensor_root_2)
                print(f"[TSF-85][INFO] Sensor 2 enabled: {sensor_root_2}")
            else:
                # Cleared -- drop the channel so single-sensor mode resumes
                # next time files are opened.
                if self._sensor2 is not None:
                    print("[TSF-85][INFO] Sensor 2 disabled.")
                self._sensor2 = None

        out_dir = s.get(f"{prefix}/output_dir")
        out_dir = str(out_dir) if out_dir else None
        if out_dir and out_dir != self._last_settings_out_dir and (initial or self._can_edit_output()):
            try:
                self._output_dir = Path(out_dir).expanduser()
                self._last_settings_out_dir = out_dir
            except Exception:
                pass

        base_name = s.get(f"{prefix}/base_name")
        base_name = str(base_name) if base_name else None
        if base_name and base_name != self._last_settings_base_name and (initial or self._can_edit_output()):
            try:
                self._base_name = self._sanitize_base_name(base_name)
                self._last_settings_base_name = base_name
            except Exception:
                pass

        # Per-output enable flags. These mirror the three GUI checkboxes
        # (Deformation maps / Synthetic data / Simulation data) but are
        # also overridable from a Python launcher script via carb settings.
        # If a setting is unset (i.e. s.get returns None), the current
        # state (GUI checkbox or constructor default) is preserved.
        for key, attr in (("log_dz",   "_log_dz_enabled"),
                          ("log_pred", "_log_pred_enabled"),
                          ("log_mesh", "_log_mesh_enabled")):
            v = s.get(f"{prefix}/{key}")
            if v is not None:
                setattr(self, attr, bool(v))

        try:
            if self._window is not None and self._window.visible:
                if hasattr(self, "_out_dir_label") and self._out_dir_label is not None:
                    self._out_dir_label.text = f"Directory: {self._output_dir}"
                if hasattr(self, "_files_preview") and self._files_preview is not None:
                    self._files_preview.text = self._files_preview_text()
        except Exception:
            pass

    # =========================================================
    # TIMELINE HELPERS
    # =========================================================
    def _is_playing(self) -> bool:
        try:
            return bool(self._timeline.is_playing())
        except Exception:
            return False

    def _time_sec(self) -> float:
        try:
            return float(self._timeline.get_current_time())
        except Exception:
            return 0.0

    def _fps(self) -> float:
        try:
            fps = float(self._timeline.get_time_codes_per_second())
            if fps > 1e-6:
                return fps
        except Exception:
            pass
        return 60.0

    def _frame_index(self, t_sec: float) -> int:
        fps = self._fps()
        return int(np.floor(t_sec * fps + 1e-9))

    # =========================================================
    # OUTPUT PATH BUILDING
    # =========================================================
    def _sanitize_base_name(self, s: str) -> str:
        if s is None:
            return "TactileData"
        s = str(s).strip()
        if s.lower().endswith(".csv"):
            s = s[:-4]
        s = s.replace("/", "_").replace("\\", "_")
        s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", s).strip("_")
        return s if s else "TactileData"

    def _compute_output_paths(self, tag: str = ""):
        """Compute the three CSV output paths.

        tag is an optional infix inserted between the base name and the
        suffix. When sensor 2 is active we call this with tag='s1' for
        sensor 1 and tag='s2' for sensor 2, producing files like
        TactileData_s1_deformations.csv. With tag='' the original
        single-sensor filenames are produced (TactileData_deformations.csv)
        to preserve backwards compatibility.
        """
        base = self._sanitize_base_name(self._base_name)
        out_dir = Path(self._output_dir).expanduser()
        infix = f"_{tag}" if tag else ""
        dz_path = out_dir / f"{base}{infix}{SUFFIX_DZ}"
        pred_path = out_dir / f"{base}{infix}{SUFFIX_PRED}"
        mesh_path = out_dir / f"{base}{infix}{SUFFIX_MESH}"
        return dz_path, pred_path, mesh_path

    def _sensor1_tag(self):
        """Tag for sensor 1's files: empty in single-sensor mode (so
        TactileData_deformations.csv etc. stay as-is), 's1' when sensor 2
        is also active. This is the single place that decides the infix."""
        return "s1" if self._sensor2 is not None else ""

    # =========================================================
    # LOAD: SENSOR CONFIG  (preferred: sensor_config.json)
    # =========================================================
    SUPPORTED_CONFIG_MAJOR = 1   # bump if the schema breaks compat

    def _load_sensor_config(self):
        """Load the single sensor_config.json bundle.
        Returns (nodes_ids, max_val) on success and raises ValueError on
        any validation failure. The error message names the exact problem
        so users can fix the config file without grepping source.
        """
        cfg_path = sensor_config_path(self._ext_root)
        if not cfg_path.exists():
            raise FileNotFoundError(str(cfg_path))

        with open(str(cfg_path), "r") as fh:
            data = json.load(fh)

        if not isinstance(data, dict):
            raise ValueError("sensor_config.json must be a JSON object at top level.")

        # Schema version check (major version only; minor is forward-compatible)
        sv = data.get("schema_version")
        if not isinstance(sv, str):
            raise ValueError("sensor_config.json missing 'schema_version' (string).")
        try:
            major = int(sv.split(".")[0])
        except Exception:
            raise ValueError(f"sensor_config.json has malformed schema_version: {sv!r}")
        if major != self.SUPPORTED_CONFIG_MAJOR:
            raise ValueError(
                f"sensor_config.json schema_version {sv!r} is not supported by this "
                f"extension build (expected major version {self.SUPPORTED_CONFIG_MAJOR})."
            )

        # Grid shape MUST match config.py exactly
        grid = data.get("grid", {})
        rows_in = grid.get("rows")
        cols_in = grid.get("cols")
        if not (isinstance(rows_in, int) and isinstance(cols_in, int)):
            raise ValueError("sensor_config.json 'grid.rows' and 'grid.cols' must be integers.")
        if rows_in != ROWS or cols_in != COLS:
            raise ValueError(
                f"sensor_config.json grid is {rows_in}x{cols_in} but the extension "
                f"expects {ROWS}x{COLS} (set in config.py). Re-export the config or "
                f"update config.py before loading."
            )

        # Node IDs: must be a 2D array of shape (ROWS, COLS), every entry a
        # non-negative int. The 2D layout is the source of truth for the
        # physical row/col arrangement; no flatten ordering convention to
        # remember.
        raw_ids = data.get("node_ids")
        if not isinstance(raw_ids, list) or len(raw_ids) != ROWS:
            raise ValueError(
                f"sensor_config.json 'node_ids' must be a list of {ROWS} rows."
            )
        flat = []
        for r, row in enumerate(raw_ids):
            if not isinstance(row, list) or len(row) != COLS:
                raise ValueError(
                    f"sensor_config.json node_ids row {r} must be a list of {COLS} ids."
                )
            for c, v in enumerate(row):
                if not isinstance(v, int) or v < 0:
                    raise ValueError(
                        f"sensor_config.json node_ids[{r}][{c}] is not a non-negative int: {v!r}"
                    )
                flat.append(v)

        # Normalization
        norm = data.get("normalization", {})
        the_max = norm.get("the_max") if isinstance(norm, dict) else None
        if not isinstance(the_max, (int, float)):
            raise ValueError("sensor_config.json missing 'normalization.the_max' (number).")
        the_max = float(the_max)
        import math
        if not math.isfinite(the_max) or the_max <= 0.0:
            raise ValueError(
                f"sensor_config.json 'normalization.the_max' must be a positive finite "
                f"number; got {the_max!r}."
            )

        return flat, the_max

    def _load_sensor_data(self):
        """Load sensor_config.json. This is the only supported config
        format; there is no fallback. Populates self._nodes_ids and
        self._max_val. On any failure both are cleared and a
        [TSF-85][ERROR] is printed; the extension stays loaded but
        inference / dz won't run until the config is fixed.
        """
        cfg_path = sensor_config_path(self._ext_root)

        if not cfg_path.exists():
            # Silent: gating will keep the extension inert. Surfaced only
            # if the user actively asks (via _asset_status or panel poll).
            self._nodes_ids = []
            self._max_val = None
            return

        try:
            ids, mv = self._load_sensor_config()
            self._nodes_ids = ids
            self._max_val = mv
            print("[TSF-85][INFO] Sensor config loaded.")
        except Exception as e:
            # Real corruption: do log this, since it usually means a bad edit.
            print(f"[TSF-85][ERROR] sensor_config.json invalid: {e}")
            self._nodes_ids = []
            self._max_val = None

    # =========================================================
    # LOAD: ONNX MODEL
    # =========================================================
    def _load_onnx_model(self):
        mp = onnx_model_path(self._ext_root)
        if not mp.exists():
            # Silent: gating will keep the extension inert.
            self._session = None
            return

        # CUDA provider options.
        # The default cudnn_conv_algo_search is "EXHAUSTIVE", which uses the
        # cuDNN frontend heuristic engine. On cuDNN 9.x with some Conv shapes
        # (e.g. small spatial maps with many channels) this engine returns
        # HEURISTIC_QUERY_FAILED and ORT falls the whole session back to CPU.
        # "HEURISTIC" routes through the older, more permissive kernel set,
        # which handles the failing shapes correctly.
        cuda_options = {
            "device_id": 0,
            "cudnn_conv_algo_search": "HEURISTIC",
            "cudnn_conv_use_max_workspace": "1",
            "do_copy_in_default_stream": "1",
        }
        providers = [
            ("CUDAExecutionProvider", cuda_options),
            "CPUExecutionProvider",
        ]

        # Session options: disable graph optimizations.
        # ORT's optimizer fuses adjacent Pad + Conv nodes back into a single
        # Conv with asymmetric padding, which then trips the cuDNN frontend
        # bug we patched the ONNX file to avoid. Disabling optimizations
        # forces ORT to keep Pad and Conv separate, so cuDNN only sees
        # symmetric-padded Conv nodes.
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        try:
            self._session = ort.InferenceSession(
                str(mp), sess_options=sess_options, providers=providers
            )
            active = self._session.get_providers()
            self._infer_input_name = self._session.get_inputs()[0].name
            # Clear cached channels-first flag so it's re-detected on the
            # next inference call (handles model swaps without a restart).
            if hasattr(self, "_infer_channels_first"):
                delattr(self, "_infer_channels_first")
            print("[TSF-85][INFO] Prediction model loaded.")

            # Explicitly warn if CUDA didn't stick. ORT silently demotes to CPU
            # when the CUDA provider fails to initialize for any reason.
            if "CUDAExecutionProvider" not in active:
                print("[TSF-85][WARN] CUDA provider not active. Inference will run on CPU.")
        except Exception as e:
            print("[TSF-85][ERROR] Failed to load ONNX model:", e)
            self._session = None

    # =========================================================
    # LOGGING DATA: START/STOP/FLUSH
    # =========================================================
    def _start_logging(self):
        if self._logging_active:
            return

        # Sensor 1's filenames depend on whether sensor 2 is active.
        dz_path, pred_path, mesh_path = self._compute_output_paths(self._sensor1_tag())
        self._dz_path = dz_path
        self._pred_path = pred_path
        self._mesh_path = mesh_path

        try:
            Path(self._output_dir).expanduser().mkdir(parents=True, exist_ok=True)

            # ---- DZ FILE (deformation maps) ----
            if self._log_dz_enabled:
                self._dz_fh = self._dz_path.open("w", newline="")
                self._dz_writer = csv.writer(self._dz_fh)
                if self._dz_path.stat().st_size == 0:
                    self._dz_writer.writerow(["time_sec", "frame"] + [f"dz_{i}" for i in range(EXPECTED_SIZE)])
                    self._dz_fh.flush()
                print(f"[TSF-85][LOG] Logging started -> {self._dz_path}")

            # ---- PRED FILE (synthetic data) ----
            if self._log_pred_enabled:
                self._pred_fh = self._pred_path.open("w", newline="")
                self._pred_writer = csv.writer(self._pred_fh)
                if self._pred_path.stat().st_size == 0:
                    self._pred_writer.writerow(["time_sec", "frame"] + [f"pred_{i}" for i in range(PRED_SIZE)])
                    self._pred_fh.flush()
                print(f"[TSF-85][LOG] Logging started -> {self._pred_path}")

            # ---- MESH STATE FILE ----
            # Long format: one row per node per frame, tab-delimited, matching
            # the schema produced by Curobo_recognizing_objects.py so the same
            # downstream tooling (e.g. Predict_tactile_slider.py) can consume
            # both. All mesh nodes are written, not just the 400 filtered ones.
            #
            # Columns:
            #   frame, t, node_id,
            #   s1_x,  s1_y,  s1_z,        # deformed position from USD
            #   s1_vx, s1_vy, s1_vz,       # velocity (0 if unavailable)
            #   s1_Rx, s1_Ry, s1_Rz,       # rest position from USD
            #   s1_Trans_x..z,             # case prim translation
            #   s1_Ori_w, s1_Ori_x..z      # case prim quaternion
            if self._log_mesh_enabled:
                self._mesh_fh = self._mesh_path.open("w", newline="")
                # Tab-delimited to match the upstream sim recorder.
                self._mesh_writer = csv.writer(self._mesh_fh, delimiter="\t")
                if self._mesh_path.stat().st_size == 0:
                    mesh_header = [
                        "frame", "t", "node_id",
                        "s1_x", "s1_y", "s1_z",
                        "s1_vx", "s1_vy", "s1_vz",
                        "s1_Rx", "s1_Ry", "s1_Rz",
                        "s1_Trans_x", "s1_Trans_y", "s1_Trans_z",
                        "s1_Ori_w", "s1_Ori_x", "s1_Ori_y", "s1_Ori_z",
                    ]
                    self._mesh_writer.writerow(mesh_header)
                    self._mesh_fh.flush()
                print(f"[TSF-85][LOG] Logging started -> {self._mesh_path}")

            # Even when no CSV is enabled we flip the session flag so the
            # rest of the physics step (dz compute, inference, heatmap) runs.
            self._logging_active = True
            self._since_flush = 0
            self._pending_close = False

            if not any([self._log_dz_enabled, self._log_pred_enabled, self._log_mesh_enabled]):
                print("[TSF-85][INFO] All CSV outputs disabled. Inference + heatmap only.")

            # If sensor 2 is enabled, open its three files in parallel.
            if self._sensor2 is not None:
                self._open_channel_files(self._sensor2)
        except Exception as e:
            print("[TSF-85][ERROR] Could not start logging:", e)
            self._close_files_now()

    def _cleanup_heatmap_pngs(self):
        """Delete the two heatmap PNG buffers if present. Called at startup
        (to wipe stale frames from a previous session) and after the timeline
        stops + inference drains (so a finished run leaves nothing behind)."""
        for p in (getattr(self, "_heatmap_png_a", None),
                  getattr(self, "_heatmap_png_b", None)):
            if p is None:
                continue
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        # Reset the toggle and force the UI to drop its reference next time
        # it tries to display, so we don't keep showing a now-deleted PNG.
        self._heatmap_png_toggle = False
        self._last_heatmap_update_t = -1e9
        try:
            if getattr(self, "_heatmap_img", None) is not None:
                self._heatmap_img.source_url = ""
        except Exception:
            pass

    # ---- Channel file helpers (used for sensor 2) ----
    # Sensor 1 keeps using the existing self._dz_fh/_pred_fh/_mesh_fh; the
    # functions below operate on a SensorChannel's own file handles.
    def _open_channel_files(self, channel):
        """Open the three CSV files for a channel (dz, pred, mesh). The
        files use the channel's tag as a filename infix. Respects the same
        per-output enable flags (_log_dz_enabled etc.) as sensor 1.
        """
        try:
            dz_path, pred_path, mesh_path = self._compute_output_paths(channel.tag)
            channel.dz_path = dz_path
            channel.pred_path = pred_path
            channel.mesh_path_csv = mesh_path

            Path(self._output_dir).expanduser().mkdir(parents=True, exist_ok=True)

            if self._log_dz_enabled:
                channel.dz_fh = channel.dz_path.open("w", newline="")
                channel.dz_writer = csv.writer(channel.dz_fh)
                if channel.dz_path.stat().st_size == 0:
                    channel.dz_writer.writerow(
                        ["time_sec", "frame"] + [f"dz_{i}" for i in range(EXPECTED_SIZE)]
                    )
                    channel.dz_fh.flush()
                print(f"[TSF-85][LOG] Logging started -> {channel.dz_path}")

            if self._log_pred_enabled:
                channel.pred_fh = channel.pred_path.open("w", newline="")
                channel.pred_writer = csv.writer(channel.pred_fh)
                if channel.pred_path.stat().st_size == 0:
                    channel.pred_writer.writerow(
                        ["time_sec", "frame"] + [f"pred_{i}" for i in range(PRED_SIZE)]
                    )
                    channel.pred_fh.flush()
                print(f"[TSF-85][LOG] Logging started -> {channel.pred_path}")

            if self._log_mesh_enabled:
                channel.mesh_fh = channel.mesh_path_csv.open("w", newline="")
                channel.mesh_writer = csv.writer(channel.mesh_fh, delimiter="\t")
                if channel.mesh_path_csv.stat().st_size == 0:
                    channel.mesh_writer.writerow([
                        "frame", "t", "node_id",
                        "s1_x", "s1_y", "s1_z",
                        "s1_vx", "s1_vy", "s1_vz",
                        "s1_Rx", "s1_Ry", "s1_Rz",
                        "s1_Trans_x", "s1_Trans_y", "s1_Trans_z",
                        "s1_Ori_w", "s1_Ori_x", "s1_Ori_y", "s1_Ori_z",
                    ])
                    channel.mesh_fh.flush()
                print(f"[TSF-85][LOG] Logging started -> {channel.mesh_path_csv}")
        except Exception as e:
            print(f"[TSF-85][ERROR] Could not start logging for {channel.tag}:", e)
            self._close_channel_files(channel)

    def _close_channel_files(self, channel):
        for fh in (channel.dz_fh, channel.pred_fh, channel.mesh_fh):
            try:
                if fh:
                    fh.flush()
                    fh.close()
            except Exception:
                pass
        channel.clear_writers()

    def _flush_channel_files(self, channel):
        for fh in (channel.dz_fh, channel.pred_fh, channel.mesh_fh):
            try:
                if fh:
                    fh.flush()
            except Exception:
                pass

    def _close_files_now(self):
        try:
            if self._dz_fh:
                self._dz_fh.flush()
                self._dz_fh.close()
        except Exception:
            pass
        try:
            if self._pred_fh:
                self._pred_fh.flush()
                self._pred_fh.close()
        except Exception:
            pass
        try:
            if self._mesh_fh:
                self._mesh_fh.flush()
                self._mesh_fh.close()
        except Exception:
            pass

        self._dz_fh = None
        self._dz_writer = None
        self._pred_fh = None
        self._pred_writer = None
        self._mesh_fh = None
        self._mesh_writer = None
        self._logging_active = False
        self._since_flush = 0

        # Close sensor 2's files too, if it's active.
        if self._sensor2 is not None:
            self._close_channel_files(self._sensor2)

        # Wipe the heatmap PNG buffers now that the run is fully closed.
        self._cleanup_heatmap_pngs()

    def _stop_logging(self):
        # Defer close if either sensor still has pending inference work.
        if self._infer_queue:
            self._pending_close = True
            return
        if self._sensor2 is not None and self._sensor2.infer_queue:
            # Mark sensor 2 to close once its queue drains. The shared
            # _pending_close flag handles sensor 1; sensor 2 has its own.
            self._sensor2.pending_close = True
            return
        self._pending_close = False
        self._close_files_now()

    def _flush_if_needed(self):
        if not self._logging_active:
            return
        self._since_flush += 1
        if self._since_flush >= self._flush_every:
            try:
                if self._dz_fh:
                    self._dz_fh.flush()
                if self._pred_fh:
                    self._pred_fh.flush()
                if self._mesh_fh:
                    self._mesh_fh.flush()
            except Exception:
                pass
            if self._sensor2 is not None:
                self._flush_channel_files(self._sensor2)
            self._since_flush = 0

    # =========================================================
    # DEFORMABLE PATH + SCHEMA  (Isaac Sim 5.1)
    # =========================================================
    def _has_any_deformable_api(self, prim) -> bool:
        """True if the prim carries any deformable schema (old PhysX or the
        new Isaac Sim 5.1 beta / auto-cooked OmniPhysics deformable)."""
        try:
            if prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI):
                return True
        except Exception:
            pass
        try:
            if prim.HasAPI(PhysxSchema.PhysxDeformableSurfaceAPI):
                return True
        except Exception:
            pass
        try:
            schemas = prim.GetAppliedSchemas()
        except Exception:
            return False
        # New beta deformable bodies apply schemas such as
        # OmniPhysicsDeformableBodyAPI, OmniPhysicsVolumeDeformableSimAPI,
        # PhysxAutoDeformableBodyAPI, and -- for auto-cooked bodies like the
        # right pad in this scene -- OmniPhysicsBodyAPI alongside an
        # auto-deformable setup. Match any of these.
        needles = ("Deformable", "OmniPhysicsBody", "PhysxAutoDeformable")
        return any(any(nd in s for nd in needles) for s in schemas)

    def _find_simulation_mesh(self, container_path: str):
        """Find the prim that holds the deformable simulation data under
        `container_path`.

        The simulation TetMesh authors a rest-points attribute
        (`omniphysics:restShapePoints`, or legacy
        `physxDeformable:simulationRestPoints`). That attribute is the most
        reliable anchor: unlike the live `points` (which at runtime can be
        served through Fabric/USDRT and read as invalid off the plain USD
        attribute), the rest attribute is authored and stable. So the finder
        keys on the rest attribute first.

        Resolution order:
          1. A prim named 'simulation_mesh' that has a rest attribute.
          2. Any prim that has a rest attribute.
          3. A prim named 'simulation_mesh' that also has `points`.
          4. Any prim with both `points` + rest (legacy direct check).
          5. First UsdGeom.Mesh under the container.
        """
        stage = self._ctx.get_stage()
        root = stage.GetPrimAtPath(container_path)
        if not root or not root.IsValid():
            return None

        def _rest_attr(prim):
            r1 = prim.GetAttribute("omniphysics:restShapePoints")
            if r1 and r1.IsValid():
                return r1
            r2 = prim.GetAttribute("physxDeformable:simulationRestPoints")
            if r2 and r2.IsValid():
                return r2
            return None

        def _has_points(prim):
            pts = prim.GetAttribute("points")
            return bool(pts and pts.IsValid())

        # 1) 'simulation_mesh' that carries a rest attribute (the real FEM
        #    sim mesh; this is what both pads use in the gripper scene).
        for prim in Usd.PrimRange(root):
            if "simulation_mesh" in prim.GetName().lower() and _rest_attr(prim) is not None:
                return prim

        # 2) Any prim with a rest attribute.
        for prim in Usd.PrimRange(root):
            if _rest_attr(prim) is not None:
                return prim

        # 3) 'simulation_mesh' that at least has points.
        for prim in Usd.PrimRange(root):
            if "simulation_mesh" in prim.GetName().lower() and _has_points(prim):
                return prim

        # 4) Any prim with both points + rest (redundant safety).
        for prim in Usd.PrimRange(root):
            if _has_points(prim) and _rest_attr(prim) is not None:
                return prim

        # 5) Fallback: first mesh under the container.
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh):
                return prim

        # Nothing matched -> dump the subtree once so we can see exactly what
        # is under this root (prim type, applied schemas, points/rest attrs).
        self._dbg_once(
            f"finder_dump_{container_path}",
            "[TSF-85][DUMP] (finder v22c) No deformable mesh matched under "
            f"{container_path}. Subtree:\n" + self._dump_subtree(root),
        )
        return None

    def _dump_subtree(self, root) -> str:
        """Return a multi-line string describing every prim under `root`:
        path, type, applied schemas, and whether it has points/rest attrs."""
        lines = []
        for prim in Usd.PrimRange(root):
            try:
                schemas = list(prim.GetAppliedSchemas())
            except Exception:
                schemas = ["<err>"]
            pts = prim.GetAttribute("points")
            has_pts = bool(pts and pts.IsValid())
            r1 = prim.GetAttribute("omniphysics:restShapePoints")
            r2 = prim.GetAttribute("physxDeformable:simulationRestPoints")
            has_rest = bool((r1 and r1.IsValid()) or (r2 and r2.IsValid()))
            lines.append(
                f"  {prim.GetPath()}  type={prim.GetTypeName()}  "
                f"points={has_pts} rest={has_rest}  schemas={schemas}"
            )
        return "\n".join(lines) if lines else "  <empty>"

    def _clear_deformable_cache(self):
        self._cached_deform_container_path = None
        self._cached_deform_mesh_path = None
        self._cached_pos_attr = None
        self._cached_rest_attr = None
        self._cached_vel_attr = None
        self._rest_attr_name = None

    def _pick_pos_attr(self, mesh_prim):
        """Choose the attribute holding current (deformed) vertex positions.

        Matches the reference SensorRecorder (_get_deformable_attrs in
        Curobo_touch_obj2.py): prefer 'physxDeformable:simulationPoints'
        (the live FEM-deformed positions PhysX populates at runtime), then
        fall back to 'points'.

        NOTE: 'deformablePose:default:omniphysics:points' is intentionally
        NOT used for positions -- in this asset it holds the static rest
        pose and never updates during simulation, which made every recorded
        frame equal to the rest shape (zero deformation). Returns (attr,
        name); attr may be None if none are present.
        """
        for name in ("physxDeformable:simulationPoints",
                     "points"):
            a = mesh_prim.GetAttribute(name)
            if a and a.IsValid():
                return a, name
        return None, None

    def _diag_n(self, key: str, n: int = 3) -> bool:
        """Return True for the first `n` times this key is seen, so a
        diagnostic can print a few times (not just once) early in a run."""
        if not hasattr(self, "_diag_counts"):
            self._diag_counts = {}
        c = self._diag_counts.get(key, 0)
        if c < n:
            self._diag_counts[key] = c + 1
            return True
        return False

    def _resolve_sensor_root(self, configured_path: str):
        """Resolve the configured sensor_root to a prim that actually exists
        on the live stage. When a robot USD is added as a reference, its
        contents can be nested one extra level deep (e.g. the path gains a
        duplicated parent segment), so the path authored in the launcher
        script may not be valid at runtime. Strategy:

          1. Use the path as-is if it resolves.
          2. Try inserting a duplicate of each path segment (handles the
             '/foo/foo/...' reference-nesting case).
          3. Search the whole stage for a prim whose path ENDS WITH the
             configured suffix (e.g. '/TSF_85_right/TSF_85') and that has a
             deformable 'simulation_mesh' under it.

        Returns the resolved path string (may equal the input), or the input
        unchanged if nothing better is found.
        """
        stage = self._ctx.get_stage()
        if stage is None:
            return configured_path

        def _ok(path):
            p = stage.GetPrimAtPath(path)
            return bool(p and p.IsValid())

        # 1) as-is
        if _ok(configured_path):
            return configured_path

        # 2) duplicate-segment variants: for each segment, try doubling it.
        segs = [s for s in configured_path.split("/") if s]
        for i in range(len(segs)):
            cand_segs = segs[:i + 1] + [segs[i]] + segs[i + 1:]
            cand = "/" + "/".join(cand_segs)
            if _ok(cand):
                self._dbg_once(
                    f"resolve_dup_{configured_path}",
                    f"[TSF-85][INFO] Resolved sensor root via segment-double: "
                    f"{configured_path} -> {cand}",
                )
                return cand

        # 3) suffix search across the whole stage.
        suffix = "/" + "/".join(segs)
        best = None
        for prim in stage.Traverse():
            ppath = prim.GetPath().pathString
            if ppath.endswith(suffix):
                # Prefer one that actually has a sim mesh under it.
                if self._find_simulation_mesh(ppath) is not None:
                    best = ppath
                    break
                if best is None:
                    best = ppath
        if best is not None:
            self._dbg_once(
                f"resolve_suffix_{configured_path}",
                f"[TSF-85][INFO] Resolved sensor root via suffix search: "
                f"{configured_path} -> {best}",
            )
            return best

        return configured_path

    def _ensure_deformable_cached(self, deform_container_path: str):
        # Resolve the configured path to whatever actually exists at runtime
        # (handles reference-nesting that shifts the prim deeper in the tree).
        resolved = self._resolve_sensor_root(deform_container_path)
        if resolved != deform_container_path:
            # Keep the active root in sync so case-pose reads use the same path.
            if getattr(self, "_active_sensor_root_path", None) == deform_container_path:
                self._active_sensor_root_path = resolved
            deform_container_path = resolved

        if deform_container_path != self._cached_deform_container_path:
            self._clear_deformable_cache()
            self._cached_deform_container_path = deform_container_path

        if self._cached_pos_attr is not None and self._cached_pos_attr.IsValid():
            return True

        # ---- Loud runtime diagnostic (first few steps) ----------------
        if self._diag_n("ensure_diag"):
            try:
                stage = self._ctx.get_stage()
                root = stage.GetPrimAtPath(deform_container_path)
                valid = bool(root and root.IsValid())
                print(f"[TSF-85][DIAG] ensure_cached path={deform_container_path} "
                      f"root_valid={valid}")
                if valid:
                    kids = [c.GetName() for c in root.GetChildren()]
                    print(f"[TSF-85][DIAG]   children={kids}")
                    found = self._find_simulation_mesh(deform_container_path)
                    print(f"[TSF-85][DIAG]   finder -> "
                          f"{found.GetPath() if found else None}")
                    if found is not None:
                        pa, pn = self._pick_pos_attr(found)
                        print(f"[TSF-85][DIAG]   pos_attr={pn} valid={bool(pa and pa.IsValid())}")
            except Exception as e:
                print(f"[TSF-85][DIAG] ensure diag error: {e}")

        mesh_prim = self._find_simulation_mesh(deform_container_path)
        if mesh_prim is None:
            return False

        self._cached_deform_mesh_path = mesh_prim.GetPath().pathString

        # --- Isaac Sim 5.1 new deformable schema attribute names ---
        # Current points: prefer the live deformable-pose attribute
        # (`deformablePose:default:omniphysics:points`), which carries the
        # simulated/deformed vertex positions at runtime. Fall back to the
        # plain `points` attribute (authored mesh) if the pose attr is absent.
        pos_attr, pos_name = self._pick_pos_attr(mesh_prim)
        vel_attr = mesh_prim.GetAttribute("velocities")

        # Rest points: new schema uses 'omniphysics:restShapePoints';
        # legacy assets used 'physxDeformable:simulationRestPoints'.
        rest_attr = mesh_prim.GetAttribute("omniphysics:restShapePoints")
        rest_name = "omniphysics:restShapePoints"
        if not (rest_attr and rest_attr.IsValid()):
            legacy = mesh_prim.GetAttribute("physxDeformable:simulationRestPoints")
            if legacy and legacy.IsValid():
                rest_attr = legacy
                rest_name = "physxDeformable:simulationRestPoints"

        if not (pos_attr and pos_attr.IsValid()):
            return False

        self._cached_pos_attr = pos_attr
        self._cached_rest_attr = rest_attr if rest_attr and rest_attr.IsValid() else None
        self._cached_vel_attr = vel_attr if vel_attr and vel_attr.IsValid() else None
        self._rest_attr_name = rest_name if self._cached_rest_attr is not None else None

        self._dbg_once(
            "deform_schema",
            f"[TSF-85][INFO] Deformable mesh: {self._cached_deform_mesh_path} | "
            f"pos attr: {pos_name} | rest attr: {self._rest_attr_name}",
        )
        return True

    def _read_deformable_arrays(self):
        tc = Usd.TimeCode.Default()
        # Re-pick the position attribute live each frame: at the moment the
        # mesh was first cached, physxDeformable:simulationPoints may not have
        # existed yet (PhysX creates it once simulation runs). Prefer it now if
        # it's present, else use the cached handle.
        pos_attr = self._cached_pos_attr
        if self._cached_deform_mesh_path:
            stage = self._ctx.get_stage()
            mp = stage.GetPrimAtPath(self._cached_deform_mesh_path) if stage else None
            if mp and mp.IsValid():
                live, _ = self._pick_pos_attr(mp)
                if live is not None and live.IsValid():
                    pos_attr = live
                    self._cached_pos_attr = live
        sim_pts = pos_attr.Get(tc) if pos_attr else None
        rest_pts = self._cached_rest_attr.Get(tc) if self._cached_rest_attr else None
        return sim_pts, rest_pts

    # ---- Channel-aware variants (used for sensor 2) ----
    # These mirror the sensor-1 cache logic above but write to a
    # SensorChannel's attributes instead of self._cached_*. Kept separate
    # so sensor 1's code path is unchanged.
    def _ensure_channel_deformable_cached(self, channel) -> bool:
        if channel.root_path is None:
            return False
        # Resolve the configured path to whatever exists at runtime (handles
        # reference-nesting), and persist it so pose reads use the same path.
        resolved = self._resolve_sensor_root(channel.root_path)
        if resolved != channel.root_path:
            channel.root_path = resolved
            channel.clear_cache()
        # Re-resolve if the cached mesh path doesn't match the root, or
        # if the cached pos_attr has gone invalid.
        if channel.pos_attr is None or not channel.pos_attr.IsValid():
            channel.clear_cache()

        if channel.pos_attr is not None and channel.pos_attr.IsValid():
            return True

        mesh_prim = self._find_simulation_mesh(channel.root_path)
        if mesh_prim is None:
            return False

        channel.mesh_path = mesh_prim.GetPath().pathString
        pos_attr, pos_name = self._pick_pos_attr(mesh_prim)
        vel_attr = mesh_prim.GetAttribute("velocities")
        rest_attr = mesh_prim.GetAttribute("omniphysics:restShapePoints")
        rest_name = "omniphysics:restShapePoints"
        if not (rest_attr and rest_attr.IsValid()):
            legacy = mesh_prim.GetAttribute("physxDeformable:simulationRestPoints")
            if legacy and legacy.IsValid():
                rest_attr = legacy
                rest_name = "physxDeformable:simulationRestPoints"

        if not (pos_attr and pos_attr.IsValid()):
            return False

        channel.pos_attr = pos_attr
        channel.rest_attr = rest_attr if rest_attr and rest_attr.IsValid() else None
        channel.vel_attr = vel_attr if vel_attr and vel_attr.IsValid() else None
        channel.rest_attr_name = rest_name if channel.rest_attr is not None else None

        self._dbg_once(
            f"deform_schema_{channel.tag}",
            f"[TSF-85][INFO] Deformable mesh ({channel.tag}): {channel.mesh_path} | "
            f"pos attr: {pos_name} | rest attr: {channel.rest_attr_name}",
        )
        return True

    def _read_channel_deformable_arrays(self, channel):
        tc = Usd.TimeCode.Default()
        pos_attr = channel.pos_attr
        if channel.mesh_path:
            stage = self._ctx.get_stage()
            mp = stage.GetPrimAtPath(channel.mesh_path) if stage else None
            if mp and mp.IsValid():
                live, _ = self._pick_pos_attr(mp)
                if live is not None and live.IsValid():
                    pos_attr = live
                    channel.pos_attr = live
        sim_pts = pos_attr.Get(tc) if pos_attr else None
        rest_pts = channel.rest_attr.Get(tc) if channel.rest_attr else None
        return sim_pts, rest_pts

    # =========================================================
    # SENSOR CASE POSE  (needed for the rest-point world transform)
    # =========================================================
    def _resolve_case_prim(self, sensor_root: str):
        """Find the sensor case prim used for the world transform of rest points.
        Looks for a child named like 'Case' under the sensor root."""
        stage = self._ctx.get_stage()
        root = stage.GetPrimAtPath(sensor_root)
        if not root or not root.IsValid():
            return None
        for prim in Usd.PrimRange(root):
            name = prim.GetName().lower()
            if name.startswith("case") or "case_m" in name:
                return prim
        # fallback: the sensor root itself
        return root

    def _read_case_pose(self, sensor_root: str):
        """Return (R_mat 3x3, T 3,) from the sensor case prim's xform ops.
        Mirrors get_dz / save_sponge_data in Test_pushing.py."""
        case_prim = self._resolve_case_prim(sensor_root)
        if case_prim is None:
            return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

        t_attr = case_prim.GetAttribute("xformOp:translate")
        o_attr = case_prim.GetAttribute("xformOp:orient")
        trans = t_attr.Get() if (t_attr and t_attr.IsValid()) else None
        ori = o_attr.Get() if (o_attr and o_attr.IsValid()) else None

        if trans is not None:
            T = np.array([float(trans[0]), float(trans[1]), float(trans[2])], dtype=np.float64)
        else:
            T = np.zeros(3, dtype=np.float64)

        if ori is not None and hasattr(ori, "GetReal"):
            ow = float(ori.GetReal())
            img = ori.GetImaginary()
            ox, oy, oz = float(img[0]), float(img[1]), float(img[2])
        else:
            ow, ox, oy, oz = 1.0, 0.0, 0.0, 0.0

        R_mat = self._rotation_matrix_np(ow, ox, oy, oz)
        return R_mat, T

    def _read_case_pose_raw(self, sensor_root: str):
        """Return (translate (3,), quat (qw,qx,qy,qz)) of the sensor case prim
        without converting to a rotation matrix. Used for raw CSV logging.
        Falls back to identity if any attribute is missing."""
        case_prim = self._resolve_case_prim(sensor_root)
        if case_prim is None:
            return np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        t_attr = case_prim.GetAttribute("xformOp:translate")
        o_attr = case_prim.GetAttribute("xformOp:orient")
        trans = t_attr.Get() if (t_attr and t_attr.IsValid()) else None
        ori = o_attr.Get() if (o_attr and o_attr.IsValid()) else None

        if trans is not None:
            T = np.array([float(trans[0]), float(trans[1]), float(trans[2])], dtype=np.float64)
        else:
            T = np.zeros(3, dtype=np.float64)

        if ori is not None and hasattr(ori, "GetReal"):
            ow = float(ori.GetReal())
            img = ori.GetImaginary()
            ox, oy, oz = float(img[0]), float(img[1]), float(img[2])
        else:
            ow, ox, oy, oz = 1.0, 0.0, 0.0, 0.0

        q = np.array([ow, ox, oy, oz], dtype=np.float64)
        return T, q

    # =========================================================
    # DEFORMATION COMPUTE  (Robotiq sensor)
    #   P_pred = P_rest @ R.T + T          # rest rotated + translated into world
    #   P_act  = curr                      # deformed points as-is
    #   value  = (P_pred - P_act)[:, DEFORM_AXIS]   # uniform pred - act
    # NO cube normalization (v23): only rotation + translation are applied.
    # DEFORM_AXIS = 0 (X).
    # =========================================================
    @staticmethod
    def _rotation_matrix_np(w, x, y, z):
        q = np.array([w, x, y, z], dtype=float)
        n = np.linalg.norm(q)
        if n < 1e-12:
            return np.eye(3, dtype=float)
        q /= n
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x*y - z*w),       2 * (x*z + y*w)],
            [2 * (x*y + z*w),       1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)],
            [2 * (x*z - y*w),       2 * (y*z + x*w),       1 - 2 * (x**2 + y**2)],
        ])

    # Axis along which deformation is measured. 0 = X, 1 = Y, 2 = Z.
    DEFORM_AXIS = 0

    def _compute_dz_from_arrays(self, rest_pts, curr_pts, R_mat, T, ordered_ids, all_node_ids):
        """Per-node deformation along DEFORM_AXIS (Y).

        rest_pts, curr_pts: (N_mesh, 3) arrays in mesh-local frame from USD
                            (`omniphysics:restShapePoints` and `points`).
        R_mat, T          : case prim's pose (mesh-local -> world).
        ordered_ids       : the sensor-node ids from sensor_config.json,
                            in row-major order.
        all_node_ids      : the full mesh node-id list (typically range(N_mesh)).

        Returns a flat (EXPECTED_SIZE,) array of deformation values, in the
        same node order as `ordered_ids`. Sign convention: pred - act for
        both pads.

        v23: only rotation + translation are applied to the rest cloud; there
        is NO cube normalization.
        """
        # Filter to just the sensor nodes, in the row-major order from JSON.
        node_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}
        filter_idx = [node_to_idx[nid] for nid in ordered_ids if nid in node_to_idx]

        rest_f = rest_pts[filter_idx]   # (N_sensor, 3) mesh-local rest
        curr_f = curr_pts[filter_idx]   # (N_sensor, 3) mesh-local current

        # Bring rest into world via the case pose: P_pred = P_rest @ R.T + T.
        # P_act is the deformed mesh point as-is from USD points.
        P_pred = (R_mat @ rest_f.T).T + T
        P_act  = curr_f

        # pred - act along DEFORM_AXIS (X), for both pads. No scaling.
        return (P_pred - P_act)[:, self.DEFORM_AXIS]

    def _dz_to_grid(self, dz_array, flip=False):
        """Reshape the flat deformation array to (ROWS, COLS). When `flip` is
        True (RIGHT pad), mirror columns (np.fliplr) to match the offline
        script's RIGHT-pad orientation."""
        arr = np.asarray(dz_array, dtype=np.float32)
        if arr.size != EXPECTED_SIZE:
            return None
        grid = arr.reshape((ROWS, COLS))
        if flip:
            grid = np.fliplr(grid)
        return grid

    # =========================================================
    # HEATMAP
    # =========================================================
    def _reset_heatmap_state(self):
        self._last_heatmap_update_t = -1e9
        self._heatmap_png_toggle = False
        try:
            if self._heatmap_img is not None:
                self._heatmap_img.source_url = ""
        except Exception:
            pass

    def _jet_rgba(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 1.0)
        r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
        a = np.ones_like(r)
        rgba = np.stack([r, g, b, a], axis=-1)
        return (rgba * 255).astype(np.uint8)

    # Fixed color scale for the predicted tactile heatmap. Keeps brightness
    # comparable across frames and across runs. Values outside this range are
    # clipped (clipping is for display only; the raw values still go into the
    # CSV unchanged).
    HEATMAP_VMIN = -25.0
    HEATMAP_VMAX = 1200.0

    def _pred_to_heatmap_rgba(self, pred: np.ndarray, h: int = PRED_ROWS, w: int = PRED_COLS) -> np.ndarray:
        pred = np.asarray(pred, dtype=np.float32).reshape(-1)
        if pred.size < PRED_SIZE:
            pred = np.concatenate([pred, np.full((PRED_SIZE - pred.size,), np.nan, dtype=np.float32)], axis=0)
        pred = pred[:PRED_SIZE]
        grid = pred.reshape(h, w)

        finite = np.isfinite(grid)
        vmin = self.HEATMAP_VMIN
        vmax = self.HEATMAP_VMAX
        denom = vmax - vmin
        if denom <= 0:
            norm = np.zeros_like(grid, dtype=np.float32)
        else:
            # Substitute 0 where non-finite so the normalization math doesn't
            # produce NaN; we restore the gray overlay below for those cells.
            grid_safe = np.where(finite, grid, 0.0)
            norm = np.clip((grid_safe - vmin) / denom, 0.0, 1.0)

        rgba = self._jet_rgba(norm)
        if np.any(~finite):
            rgba[~finite] = np.array([80, 80, 80, 255], dtype=np.uint8)
        return rgba

    def _write_heatmap_png(self, rgba_small: np.ndarray, out_path: str,
                           out_w: int = 420, out_h: int = 240,
                           pred_grid: np.ndarray = None) -> bool:
        """Render the small heatmap, upscale, and (optionally) draw per-cell
        numeric values on top. pred_grid should be a (PRED_ROWS, PRED_COLS)
        float array of the raw (un-normalized) predictions; if None, no
        numbers are drawn.
        """
        if Image is None:
            self._dbg_once("no_pillow", "[TSF-85][WARN] Pillow not available; cannot write heatmap PNG.")
            return False
        try:
            img = Image.fromarray(rgba_small, mode="RGBA")
            img = img.resize((out_w, out_h), resample=Image.NEAREST)

            # Overlay per-cell values for debugging.
            if pred_grid is not None and ImageDraw is not None:
                self._overlay_cell_values(img, rgba_small, pred_grid, out_w, out_h)

            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path, format="PNG")
            return True
        except Exception as e:
            self._dbg_once("png_write_err", f"[TSF-85][WARN] Failed to write heatmap PNG: {e}")
            return False

    def _overlay_cell_values(self, img, rgba_small, pred_grid, out_w, out_h):
        """Draw the raw prediction value in the center of each cell, in either
        white or black depending on cell brightness so it stays readable on
        any jet color."""
        rows, cols = pred_grid.shape
        cell_w = out_w / cols
        cell_h = out_h / rows

        # Pick a font. Try DejaVuSans (standard on Ubuntu) at a sensible size
        # for the cell. Fall back to PIL's bitmap default if not available.
        font_size = max(10, int(min(cell_w, cell_h) * 0.28))
        font = None
        for fp in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue
        if font is None:
            try:
                font = ImageFont.load_default()
            except Exception:
                return  # No font of any kind; bail silently.

        draw = ImageDraw.Draw(img)
        for r in range(rows):
            for c in range(cols):
                v = float(pred_grid[r, c])
                if not np.isfinite(v):
                    text = "NaN"
                else:
                    text = f"{v:.2f}"

                # Cell center in upscaled image coords
                cx = (c + 0.5) * cell_w
                cy = (r + 0.5) * cell_h

                # Decide ink color: white text on dark cells, black on light.
                # Use the small RGBA's per-cell luminance.
                px = rgba_small[r, c]
                lum = 0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2]
                ink = (255, 255, 255, 255) if lum < 128 else (0, 0, 0, 255)

                # Center the text on the cell. Use textbbox for accurate sizing.
                try:
                    bbox = draw.textbbox((0, 0), text, font=font)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                except Exception:
                    tw, th = draw.textsize(text, font=font)
                draw.text((cx - tw / 2, cy - th / 2), text, fill=ink, font=font)

    def _update_heatmap_if_needed(self):
        if getattr(self, "_heatmap_img", None) is None:
            return
        if self._last_pred is None or self._last_pred_frame is None:
            return

        show = False
        if getattr(self, "_show_heatmap", None) is not None:
            try:
                show = bool(self._show_heatmap.model.get_value_as_bool())
            except Exception:
                show = False
        if not show:
            return

        now = float(self._time_sec())
        if (now - float(getattr(self, "_last_heatmap_update_t", -1e9))) < HEATMAP_MIN_DT:
            return

        rgba = self._pred_to_heatmap_rgba(self._last_pred)
        self._heatmap_png_toggle = not getattr(self, "_heatmap_png_toggle", False)
        out_path = self._heatmap_png_a if self._heatmap_png_toggle else self._heatmap_png_b

        ok = self._write_heatmap_png(rgba, str(out_path), out_w=300, out_h=600)
        if not ok:
            return

        try:
            self._heatmap_img.source_url = out_path.as_uri()
            self._last_heatmap_update_t = now
        except Exception as e:
            self._dbg_once("heatmap_set_url", f"[TSF-85][WARN] Failed to set heatmap image url: {e}")

    # =========================================================
    # INFERENCE
    # =========================================================
    def _predict_from_grid(self, grid_25x16):
        """Mirror test_retrained_cnn.py: normalize by the_max, auto-transpose
        to channels-first if the ONNX expects it, run inference."""
        if self._session is None:
            self._dbg_once("no_model", "[TSF-85][DBG] Inference disabled: ONNX model not loaded.")
            return None

        grid = np.asarray(grid_25x16, dtype=np.float32)
        if self._max_val is not None and abs(float(self._max_val)) > 1e-12:
            grid = grid / float(self._max_val)

        # Base shape: channels-last (1, ROWS, COLS, 1).
        x = grid.reshape(1, ROWS, COLS, 1).astype(np.float32)

        # Some ONNX exports use channels-first (1, 1, ROWS, COLS). Detect from
        # the model's declared input shape and transpose if needed. Cached
        # on first call so we don't introspect every frame.
        if not hasattr(self, "_infer_channels_first"):
            try:
                exp = self._session.get_inputs()[0].shape
                self._infer_channels_first = (
                    len(exp) == 4 and isinstance(exp[1], int) and exp[1] == 1
                )
                if self._infer_channels_first:
                    print("[TSF-85][INFO] ONNX is channels-first; will transpose input.")
            except Exception:
                self._infer_channels_first = False

        if self._infer_channels_first:
            x = np.transpose(x, (0, 3, 1, 2))   # (1, 1, ROWS, COLS)

        try:
            out = self._session.run(None, {self._infer_input_name: x})[0]
            return np.asarray(out).reshape(-1)
        except Exception as e:
            self._dbg_once("infer_run_err", f"[TSF-85][WARN] ONNX inference failed: {e}")
            return None

    async def _inference_loop(self):
        while getattr(self, "_running", False):
            try:
                # ---- Sensor 1 ----
                if self._infer_queue:
                    t, frame_idx, grid = self._infer_queue.pop()
                    self._infer_queue.clear()

                    pred = self._predict_from_grid(grid)
                    if pred is None:
                        await asyncio.sleep(0.001)
                        continue

                    self._last_pred = pred
                    self._last_pred_frame = frame_idx
                    self._dbg_once("pred_ok", f"[TSF-85][DBG] Predictions running (pred0={float(pred[0]):.4f})")

                    if self._record_active and self._logging_active and self._pred_writer is not None:
                        pred_n = np.asarray(pred, dtype=np.float32).reshape(-1).tolist()
                        if len(pred_n) < PRED_SIZE:
                            pred_n = pred_n + [float("nan")] * (PRED_SIZE - len(pred_n))
                        else:
                            pred_n = pred_n[:PRED_SIZE]
                        self._pred_writer.writerow([f"{t:.6f}", int(frame_idx)] + [float(v) for v in pred_n])
                        self._flush_if_needed()

                    if self._pending_close and not self._infer_queue:
                        self._pending_close = False
                        self._close_files_now()

                # ---- Sensor 2 ----
                # Shares the ONNX session with sensor 1. ORT InferenceSession
                # is internally thread-safe for run(), so calling it back-to-
                # back from the same async loop is fine -- they're serialised
                # here anyway, no actual concurrency.
                if self._sensor2 is not None and self._sensor2.infer_queue:
                    ch = self._sensor2
                    t2, frame_idx2, grid2 = ch.infer_queue.pop()
                    ch.infer_queue.clear()
                    pred2 = self._predict_from_grid(grid2)
                    if pred2 is not None:
                        ch.last_pred = pred2
                        ch.last_pred_frame = frame_idx2
                        self._dbg_once("s2_pred_ok",
                                       f"[TSF-85][DBG] Sensor 2 predictions running (pred0={float(pred2[0]):.4f})")
                        if self._record_active and self._logging_active and ch.pred_writer is not None:
                            pn = np.asarray(pred2, dtype=np.float32).reshape(-1).tolist()
                            if len(pn) < PRED_SIZE:
                                pn = pn + [float("nan")] * (PRED_SIZE - len(pn))
                            else:
                                pn = pn[:PRED_SIZE]
                            ch.pred_writer.writerow([f"{t2:.6f}", int(frame_idx2)] + [float(v) for v in pn])
                        # Drained -> if a close was deferred for sensor 2, do it now.
                        if ch.pending_close and not ch.infer_queue:
                            ch.pending_close = False
                            # Only close files if sensor 1 is also fully done.
                            if not self._pending_close and not self._infer_queue:
                                self._close_files_now()

            except Exception as e:
                self._dbg_once("infer_exc", f"[TSF-85][WARN] Inference loop error: {e}")

            await asyncio.sleep(0.001)

    # =========================================================
    # PHYSX STEP
    # =========================================================
    def _asset_status(self):
        """Quick check of whether the extension's data assets are usable.
        Returns (ok: bool, message: str). Message is shown in the UI.

        Required at runtime:
          - sensor_config.json loaded (i.e. self._nodes_ids non-empty and
            self._max_val set)
          - ONNX model loaded (i.e. self._session set)

        If either is missing, the extension cannot do anything useful;
        we refuse to start logging.
        """
        problems = []
        if not self._nodes_ids:
            problems.append("sensor_config.json not loaded")
        if self._max_val is None:
            problems.append("normalization 'the_max' not loaded")
        if self._session is None:
            problems.append("ONNX model not loaded")

        if not problems:
            return True, "Assets OK."

        cfg_path = sensor_config_path(self._ext_root)
        onnx_path = onnx_model_path(self._ext_root)
        msg = "MISSING: " + ", ".join(problems)
        msg += f"  |  expected files: {cfg_path.name} and {onnx_path.name} under {cfg_path.parent}"
        return False, msg

    def _logging_allowed(self) -> bool:
        """All of these must hold for logging/inference/heatmap to run:
        - the extension's panel window is open and visible
            (skipped in headless mode -- set /exts/TSF_85_Ext/headless = True)
        - a sensor has been selected (active_sensor_root_path is set)
        - the extension's data assets are loaded (sensor_config + ONNX)
        """
        if not self._headless:
            panel_open = self._window is not None and self._window.visible
            if not panel_open:
                return False
        if not self._active_sensor_root_path:
            return False
        ok, _msg = self._asset_status()
        return ok

    def _on_physics_step(self, dt: float):
        if not self._is_playing():
            return

        # GATING: do not start any logging, inference, or heatmap updates
        # unless the panel is open AND a sensor has been selected AND the
        # extension's assets (config + model) are loaded.
        if not self._logging_allowed():
            # If logging happens to be active from a previous valid state
            # (e.g. user closed the panel or deselected mid-run), close cleanly.
            if self._logging_active:
                self._stop_logging()
            return

        if not self._logging_active:
            self._start_logging()

        t = self._time_sec()
        frame_idx = self._frame_index(t)

        # Refresh the record gate from carb each step (script toggles it).
        prev_rec = self._record_active
        self._record_active = bool(
            self._settings.get(f"{self._settings_prefix}/record_active") or False)
        if self._record_active != prev_rec:
            state = "ON" if self._record_active else "OFF"
            print(f"[TSF-85][REC] record_active -> {state} at frame {frame_idx}")

        if self._last_processed_frame is not None and frame_idx == self._last_processed_frame:
            return
        self._last_processed_frame = frame_idx

        if not self._active_sensor_root_path:
            self._dbg_once("no_sensor",
                           "[TSF-85][DBG] No sensor_root set. Set "
                           f"{self._settings_prefix}/sensor_root or pick in UI.")
            return

        if not self._ensure_deformable_cached(self._active_sensor_root_path):
            self._dbg_once("cache_fail",
                           f"[TSF-85][DBG] Could not find a deformable simulation mesh under "
                           f"{self._active_sensor_root_path}")
            return

        sim_pts, rest_pts = self._read_deformable_arrays()
        if sim_pts is None or rest_pts is None:
            self._dbg_once("pts_none",
                           f"[TSF-85][DBG] sim/rest points missing on {self._cached_deform_mesh_path}")
            return

        n = len(sim_pts)
        if len(rest_pts) != n:
            self._dbg_once("len_mismatch",
                           f"[TSF-85][DBG] points/restPoints length mismatch: {n} vs {len(rest_pts)}")
            return

        # Build numpy arrays
        curr_np = np.array([[float(sim_pts[i][0]), float(sim_pts[i][1]), float(sim_pts[i][2])]
                            for i in range(n)], dtype=np.float64)
        rest_np = np.array([[float(rest_pts[i][0]), float(rest_pts[i][1]), float(rest_pts[i][2])]
                            for i in range(n)], dtype=np.float64)

        # Diagnostic: confirm the recorded positions are actually deforming
        # (i.e. sim points differ from rest). Prints a few times early in the
        # run. If max|sim-rest| stays ~0, the wrong (static) attribute is being
        # read.
        if self._diag_n("deform_diag", n=5):
            d = float(np.max(np.abs(curr_np - rest_np)))
            pa_name = getattr(self._cached_pos_attr, "GetName", lambda: "?")()
            print(f"[TSF-85][DIAG] pos_attr={pa_name}  max|sim-rest|={d:.6f}  "
                  f"(>0 means deformation is being captured)")

        # Sensor case pose -> world transform for rest points
        R_mat, T = self._read_case_pose(self._active_sensor_root_path)

        # Filter to the CSV node ids (must yield EXPECTED_SIZE nodes)
        ordered_ids = [i for i in self._nodes_ids if isinstance(i, int) and 0 <= i < n]
        if len(ordered_ids) != EXPECTED_SIZE:
            self._dbg_once("idx_mismatch",
                           f"[TSF-85][DBG] Node ids mismatch: {len(ordered_ids)} != {EXPECTED_SIZE}")
            return

        try:
            dz = self._compute_dz_from_arrays(rest_np, curr_np, R_mat, T,
                                              ordered_ids, list(range(n)))
        except Exception as e:
            self._dbg_once("dz_err", f"[TSF-85][WARN] dz computation failed: {e}")
            return

        dz = np.asarray(dz, dtype=np.float32).reshape(-1)
        if dz.size != EXPECTED_SIZE:
            self._dbg_once("dz_size", f"[TSF-85][DBG] dz size {dz.size} != {EXPECTED_SIZE}")
            return

        # Sensor 1 = RIGHT pad: column-mirror (fliplr) the flat dz so the flip
        # propagates everywhere downstream (CSV write AND CNN grid). Done on
        # the flat array here rather than only on the grid, per option B.
        dz = np.fliplr(dz.reshape((ROWS, COLS))).reshape(-1)

        grid = self._dz_to_grid(dz)
        if grid is None:
            self._dbg_once("grid_none", "[TSF-85][DBG] grid reshape failed")
            return

        self._last_dz = dz.tolist()
        self._last_time = t
        self._last_frame = frame_idx

        if self._record_active and self._dz_writer is not None:
            self._dz_writer.writerow([f"{t:.6f}", int(frame_idx)] + [float(v) for v in dz])
            self._flush_if_needed()

        # ---- MESH STATE ROWS (long format) ----
        # One row per node per frame, all mesh nodes (not just the filtered
        # sensor 400). Matches the schema from Curobo_recognizing_objects.py
        # minus the object/phase columns.
        if self._record_active and self._mesh_writer is not None:
            try:
                case_T, case_q = self._read_case_pose_raw(self._active_sensor_root_path)

                # Velocities are optional. If the deformable doesn't expose
                # them, write zeros (same fallback as the upstream recorder).
                vel_pts = None
                if self._cached_vel_attr is not None:
                    try:
                        vel_pts = self._cached_vel_attr.Get(Usd.TimeCode.Default())
                    except Exception:
                        vel_pts = None

                n_mesh = len(sim_pts)
                n_vel = len(vel_pts) if vel_pts is not None else 0
                n_rest = len(rest_pts)
                tstr = f"{t:.6f}"
                tx, ty, tz = float(case_T[0]), float(case_T[1]), float(case_T[2])
                qw, qx, qy, qz = (float(case_q[0]), float(case_q[1]),
                                  float(case_q[2]), float(case_q[3]))

                writer = self._mesh_writer
                for i in range(n_mesh):
                    px, py, pz = float(sim_pts[i][0]), float(sim_pts[i][1]), float(sim_pts[i][2])
                    if vel_pts is not None and i < n_vel:
                        vx = float(vel_pts[i][0]); vy = float(vel_pts[i][1]); vz = float(vel_pts[i][2])
                    else:
                        vx = vy = vz = 0.0
                    if i < n_rest:
                        rx = float(rest_pts[i][0]); ry = float(rest_pts[i][1]); rz = float(rest_pts[i][2])
                    else:
                        rx = ry = rz = 0.0
                    writer.writerow([
                        int(frame_idx), tstr, i,
                        px, py, pz,
                        vx, vy, vz,
                        rx, ry, rz,
                        tx, ty, tz,
                        qw, qx, qy, qz,
                    ])
            except Exception as e:
                self._dbg_once("mesh_row_err",
                               f"[TSF-85][WARN] mesh-state row failed: {e}")

        self._infer_queue.append((t, frame_idx, grid))

        # If sensor 2 is enabled, run the same pipeline for it in parallel.
        # Sensor 2 has its own cached mesh, its own inference queue, and
        # its own CSV writers. It shares the ONNX session and sensor_config
        # with sensor 1.
        if self._sensor2 is not None:
            try:
                self._process_sensor2_physics(t, frame_idx)
            except Exception as e:
                self._dbg_once("s2_step_err",
                               f"[TSF-85][WARN] Sensor 2 physics step failed: {e}")

    def _process_sensor2_physics(self, t, frame_idx):
        """Per-step pipeline for sensor 2. Mirrors the inline sensor-1
        block in _on_physics_step but operates on self._sensor2's channel
        state. Called only when sensor 2 is enabled."""
        ch = self._sensor2

        # Per-channel idempotency: same guard sensor 1 has at the top of
        # _on_physics_step. (The frame_idx check there protected the
        # whole step, which is enough; this is belt-and-suspenders.)
        if ch.last_processed_frame is not None and ch.last_processed_frame == frame_idx:
            return
        ch.last_processed_frame = frame_idx

        if not self._ensure_channel_deformable_cached(ch):
            self._dbg_once("s2_cache_fail",
                           f"[TSF-85][DBG] Could not find a deformable simulation mesh under {ch.root_path}")
            return

        sim_pts, rest_pts = self._read_channel_deformable_arrays(ch)
        if sim_pts is None or rest_pts is None:
            self._dbg_once("s2_pts_none",
                           f"[TSF-85][DBG] sim/rest points missing on {ch.mesh_path}")
            return

        n = len(sim_pts)
        if len(rest_pts) != n:
            self._dbg_once("s2_len_mismatch",
                           f"[TSF-85][DBG] s2 points/rest length mismatch: {n} vs {len(rest_pts)}")
            return

        # Numpy arrays
        curr_np = np.array([[float(sim_pts[i][0]), float(sim_pts[i][1]), float(sim_pts[i][2])]
                            for i in range(n)], dtype=np.float64)
        rest_np = np.array([[float(rest_pts[i][0]), float(rest_pts[i][1]), float(rest_pts[i][2])]
                            for i in range(n)], dtype=np.float64)

        # Case pose for sensor 2's case prim
        R_mat, T = self._read_case_pose(ch.root_path)

        # Filter to sensor nodes (same node_ids used for both sensors --
        # they're assumed identical hardware)
        ordered_ids = [i for i in self._nodes_ids if isinstance(i, int) and 0 <= i < n]
        if len(ordered_ids) != EXPECTED_SIZE:
            self._dbg_once("s2_idx_mismatch",
                           f"[TSF-85][DBG] s2 node ids mismatch: {len(ordered_ids)} != {EXPECTED_SIZE}")
            return

        try:
            dz = self._compute_dz_from_arrays(rest_np, curr_np, R_mat, T,
                                              ordered_ids, list(range(n)))
        except Exception as e:
            self._dbg_once("s2_dz_err", f"[TSF-85][WARN] s2 dz failed: {e}")
            return

        dz = np.asarray(dz, dtype=np.float32).reshape(-1)
        if dz.size != EXPECTED_SIZE:
            return
        grid = self._dz_to_grid(dz)
        if grid is None:
            return

        ch.last_dz = dz.tolist()
        ch.last_time = t
        ch.last_frame = frame_idx

        # Write dz row to sensor 2's CSV
        if self._record_active and ch.dz_writer is not None:
            ch.dz_writer.writerow([f"{t:.6f}", int(frame_idx)] + [float(v) for v in dz])

        # Write mesh-state rows (all nodes) to sensor 2's mesh CSV
        if self._record_active and ch.mesh_writer is not None:
            try:
                case_T, case_q = self._read_case_pose_raw(ch.root_path)
                vel_pts = None
                if ch.vel_attr is not None:
                    try:
                        vel_pts = ch.vel_attr.Get(Usd.TimeCode.Default())
                    except Exception:
                        vel_pts = None

                n_mesh = len(sim_pts)
                n_vel = len(vel_pts) if vel_pts is not None else 0
                n_rest = len(rest_pts)
                tstr = f"{t:.6f}"
                tx, ty, tz = float(case_T[0]), float(case_T[1]), float(case_T[2])
                qw, qx, qy, qz = (float(case_q[0]), float(case_q[1]),
                                  float(case_q[2]), float(case_q[3]))
                w = ch.mesh_writer
                for i in range(n_mesh):
                    px, py, pz = float(sim_pts[i][0]), float(sim_pts[i][1]), float(sim_pts[i][2])
                    if vel_pts is not None and i < n_vel:
                        vx = float(vel_pts[i][0]); vy = float(vel_pts[i][1]); vz = float(vel_pts[i][2])
                    else:
                        vx = vy = vz = 0.0
                    if i < n_rest:
                        rx = float(rest_pts[i][0]); ry = float(rest_pts[i][1]); rz = float(rest_pts[i][2])
                    else:
                        rx = ry = rz = 0.0
                    w.writerow([
                        int(frame_idx), tstr, i,
                        px, py, pz,
                        vx, vy, vz,
                        rx, ry, rz,
                        tx, ty, tz,
                        qw, qx, qy, qz,
                    ])
            except Exception as e:
                self._dbg_once("s2_mesh_row_err",
                               f"[TSF-85][WARN] s2 mesh-state row failed: {e}")

        # Queue grid for inference
        ch.infer_queue.append((t, frame_idx, grid))

    # =========================================================
    # UI
    # =========================================================
    def show_window(self, *_):
        if self._panel is None:
            self._panel = CoRoPanel(self)
        self._panel.show()

        self._window = self._panel.window
        self._out_dir_label = self._panel.out_dir_label
        self._files_preview = self._panel.files_preview
        self._sensor_label = self._panel.sensor_label
        self._mode_label = self._panel.mode_label
        self._def_nodes = self._panel.def_nodes
        self._def_mesh = self._panel.def_mesh
        self._def_dz_count = self._panel.def_dz_count
        self._def_dz_preview = self._panel.def_dz_preview
        self._pred_preview = self._panel.pred_preview
        self._show_heatmap = self._panel.show_heatmap
        self._heatmap_img = self._panel.heatmap_img
        self._log_label = self._panel.log_label
        self._out_dir_field = self._panel.out_dir_field
        self._base_name_field = self._panel.base_name_field
        self._log_dz_check = self._panel.log_dz_check
        self._log_pred_check = self._panel.log_pred_check
        self._log_mesh_check = self._panel.log_mesh_check

    def _files_preview_text(self):
        dz_path, pred_path, mesh_path = self._compute_output_paths(self._sensor1_tag())
        active = []
        if self._log_dz_enabled:
            active.append(dz_path.name)
        if self._log_pred_enabled:
            active.append(pred_path.name)
        if self._log_mesh_enabled:
            active.append(mesh_path.name)
        if not active:
            return "Files to generate:  (none \u2014 all outputs disabled)"
        if len(active) == 1:
            return f"Files to generate:  {active[0]}"
        if len(active) == 2:
            return f"Files to generate:  {active[0]} and {active[1]}"
        return f"Files to generate:  {active[0]}, {active[1]}, and {active[2]}"

    def _can_edit_output(self) -> bool:
        return (not self._is_playing()) and (not self._logging_active)

    def _set_log_flag(self, which: str, enabled: bool):
        """Called by the panel when a 'which output' checkbox is toggled.
        `which` is one of: 'dz', 'pred', 'mesh'. Edits are only accepted
        while the output settings are editable (not playing / not logging)."""
        if not self._can_edit_output():
            print("[TSF-85][WARN] Stop simulation + click 'New run' before changing outputs.")
            # Refresh the panel checkbox to reflect the unchanged state.
            self._refresh_log_checkboxes()
            return

        if which == "dz":
            self._log_dz_enabled = bool(enabled)
        elif which == "pred":
            self._log_pred_enabled = bool(enabled)
        elif which == "mesh":
            self._log_mesh_enabled = bool(enabled)
        else:
            return

        # Refresh the "Files to generate" line.
        if hasattr(self, "_files_preview") and self._files_preview is not None:
            try:
                self._files_preview.text = self._files_preview_text()
            except Exception:
                pass

    def _refresh_log_checkboxes(self):
        """Force the panel checkboxes to mirror the extension's flags
        (used after rejecting an edit)."""
        if self._panel is None:
            return
        try:
            if self._panel.log_dz_check is not None:
                self._panel.log_dz_check.model.set_value(self._log_dz_enabled)
            if self._panel.log_pred_check is not None:
                self._panel.log_pred_check.model.set_value(self._log_pred_enabled)
            if self._panel.log_mesh_check is not None:
                self._panel.log_mesh_check.model.set_value(self._log_mesh_enabled)
        except Exception:
            pass

    def _apply_output_dir(self):
        if not self._can_edit_output():
            print("[TSF-85][WARN] Stop simulation + click 'New run' before changing directory.")
            return
        try:
            raw = self._out_dir_field.model.get_value_as_string()
        except Exception:
            raw = ""
        raw = (raw or "").strip()
        if not raw:
            print("[TSF-85][WARN] Output directory is empty.")
            return
        try:
            p = Path(raw).expanduser()
            p.mkdir(parents=True, exist_ok=True)
            self._output_dir = p
        except Exception as e:
            print("[TSF-85][WARN] Invalid directory path:", raw, "|", e)
            return
        self._settings.set(f"{self._settings_prefix}/output_dir", str(self._output_dir))
        self._last_settings_out_dir = str(self._output_dir)
        if hasattr(self, "_out_dir_label") and self._out_dir_label is not None:
            self._out_dir_label.text = f"Directory: {self._output_dir}"
        if hasattr(self, "_files_preview") and self._files_preview is not None:
            self._files_preview.text = self._files_preview_text()
        print("[TSF-85][INFO] Output directory set to:", self._output_dir)

    def _apply_base_name(self):
        if not self._can_edit_output():
            print("[TSF-85][WARN] Stop simulation + click 'New run' before changing base name.")
            return
        try:
            name = self._base_name_field.model.get_value_as_string()
        except Exception:
            name = "TactileData"
        self._base_name = self._sanitize_base_name(name)
        self._settings.set(f"{self._settings_prefix}/base_name", str(self._base_name))
        self._last_settings_base_name = str(self._base_name)
        if hasattr(self, "_files_preview") and self._files_preview is not None:
            self._files_preview.text = self._files_preview_text()
        print("[TSF-85][INFO] Base name set to:", self._base_name)

    def _new_run(self):
        self._stop_logging()
        self._last_processed_frame = None
        self._reset_heatmap_state()
        print("[TSF-85][INFO] New run ready. You can change name/directory now.")

    def _pick_sensor(self):
        self._awaiting_new_sensor_pick = True
        if hasattr(self, "_mode_label") and self._mode_label is not None:
            self._mode_label.text = "Now click your sensor root once"

    def _clear_sensor(self):
        self._active_sensor_root_path = None
        self._awaiting_new_sensor_pick = True
        self._clear_deformable_cache()
        self._last_dz = None
        self._last_pred = None
        self._last_pred_frame = None
        self._last_processed_frame = None

    # =========================================================
    # POLL LOOP
    # =========================================================
    async def _poll_loop(self):
        while getattr(self, "_running", False):
            try:
                playing = self._is_playing()

                if (not self._was_playing) and playing:
                    self._reset_heatmap_state()

                self._apply_settings()

                if self._was_playing and (not playing):
                    if self._logging_active:
                        print("[TSF-85][INFO] Timeline stopped -> closing CSV files (may defer until inference drains).")
                        self._stop_logging()
                    self._last_processed_frame = None
                    self._reset_heatmap_state()

                self._was_playing = playing

                if self._window is not None and self._window.visible:
                    if self._awaiting_new_sensor_pick:
                        paths = self._sel.get_selected_prim_paths()
                        current = paths[0] if paths else None
                        if current:
                            stage = self._ctx.get_stage()
                            prim = stage.GetPrimAtPath(current)
                            if prim and prim.IsValid():
                                self._active_sensor_root_path = current
                                self._awaiting_new_sensor_pick = False
                                self._clear_deformable_cache()
                                if hasattr(self, "_mode_label") and self._mode_label is not None:
                                    self._mode_label.text = "Locked"
                                if not prim.HasAttribute(ATTR_NAME):
                                    prim.CreateAttribute(ATTR_NAME, Sdf.ValueTypeNames.String).Set(DEFAULT_VALUE)

                    if hasattr(self, "_sensor_label") and self._sensor_label is not None:
                        self._sensor_label.text = (
                            f"Active sensor: {self._active_sensor_root_path}"
                            if self._active_sensor_root_path
                            else "Active sensor: (none)"
                        )

                    if self._active_sensor_root_path:
                        if self._ensure_deformable_cached(self._active_sensor_root_path):
                            if hasattr(self, "_def_mesh") and self._def_mesh is not None:
                                self._def_mesh.text = f"Mesh: {self._cached_deform_mesh_path}"
                        else:
                            if hasattr(self, "_def_mesh") and self._def_mesh is not None:
                                self._def_mesh.text = "Mesh: (not found)"

                    if self._last_dz is not None:
                        if hasattr(self, "_def_dz_count") and self._def_dz_count is not None:
                            self._def_dz_count.text = (
                                f"Analyzed: {len(self._last_dz)} | frame={self._last_frame} | t={self._last_time:.4f}s"
                            )
                        if hasattr(self, "_def_dz_preview") and self._def_dz_preview is not None:
                            self._def_dz_preview.text = f"dZ preview: {self._preview_list(self._last_dz, k=6)}"
                    else:
                        if hasattr(self, "_def_dz_count") and self._def_dz_count is not None:
                            self._def_dz_count.text = "Analyzed: (none)"
                        if hasattr(self, "_def_dz_preview") and self._def_dz_preview is not None:
                            self._def_dz_preview.text = "dZ preview: (none)"

                    if self._last_pred is not None:
                        k = min(8, len(self._last_pred))
                        preview = ", ".join(f"{float(v):.4f}" for v in self._last_pred[:k])
                        if hasattr(self, "_pred_preview") and self._pred_preview is not None:
                            self._pred_preview.text = (
                                f"Prediction preview (frame={self._last_pred_frame}): "
                                f"[{preview}{', ...' if len(self._last_pred) > k else ''}]"
                            )
                    else:
                        if hasattr(self, "_pred_preview") and self._pred_preview is not None:
                            self._pred_preview.text = "Prediction preview: (none)"

                    self._update_heatmap_if_needed()

                    if hasattr(self, "_log_label") and self._log_label is not None:
                        if playing and self._logging_active:
                            self._log_label.text = (
                                f"Logging: ACTIVE -> "
                                f"{self._dz_path.name if self._dz_path else '(starting...)'} & "
                                f"{self._pred_path.name if self._pred_path else '(starting...)'}"
                            )
                        else:
                            self._log_label.text = "Logging: stopped (press Play to start)."

            except Exception as e:
                print("[TSF-85][WARN] UI poll error:", e)

            await asyncio.sleep(0.1)

    def _preview_list(self, arr, k=5):
        if arr is None or len(arr) == 0:
            return "(none)"
        vals = arr[:k]
        return "[" + ", ".join(f"{float(v):.6f}" for v in vals) + (", ...]" if len(arr) > k else "]")
