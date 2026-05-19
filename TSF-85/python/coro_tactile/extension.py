import omni
import omni.ext
import asyncio
import csv
from pathlib import Path
from collections import deque
from collections.abc import Mapping
import re

import numpy as np
import omni.ui as ui
import omni.usd
import omni.timeline
import omni.physx as _physx
import carb.settings
from pxr import Sdf, Usd, UsdGeom

import tensorflow as tf

try:
    from PIL import Image  # Pillow
except Exception:
    Image = None

from .config import *
from .ui_panel import CoRoPanel

class Extension(omni.ext.IExt):

    # =========================================================
    # STARTUP / SHUTDOWN
    # =========================================================
    def on_startup(self, ext_id):
        self._running = True

        self._ctx = omni.usd.get_context()
        self._sel = self._ctx.get_selection()
        self._timeline = omni.timeline.get_timeline_interface()

        # Settings (script control)
        self._settings = carb.settings.get_settings()
        self._settings_prefix = "/exts/coro_tactile"

        # Output settings defaults
        self._output_dir = DEFAULT_OUTPUT_DIR
        self._base_name = "TactileData"

        # Sensor selection defaults
        self._active_sensor_root_path = None
        self._awaiting_new_sensor_pick = True

        # UI
        self._window = None
        self._panel = None

        # Heatmap UI (file-based)
        self._heatmap_img = None
        self._heatmap_png_a = (EXT_ROOT / "data" / "_pred_heatmap_a.png").resolve()
        self._heatmap_png_b = (EXT_ROOT / "data" / "_pred_heatmap_b.png").resolve()
        self._heatmap_png_toggle = False
        self._last_heatmap_update_t = -1e9


        # Node ids
        self._nodes_ids = self._load_nodes_csv()

        # Deformable cache
        self._clear_deformable_cache()

        # Model + signature
        self._model = None
        self._infer_fn = None
        self._infer_input_name = None
        self._load_savedmodel()

        # Max value
        self._max_val = None
        self._load_max_value()

        # Latest values
        self._last_dz = None
        self._last_time = 0.0
        self._last_frame = 0
        self._last_pred = None
        self._last_pred_frame = None

        # Logging
        self._dz_path = None
        self._pred_path = None
        self._dz_fh = None
        self._dz_writer = None
        self._pred_fh = None
        self._pred_writer = None
        self._logging_active = False
        self._flush_every = 50
        self._since_flush = 0

        # Ensure one row per timeline frame
        self._last_processed_frame = None

        # Track play state to detect Stop events
        self._was_playing = False

        # Inference queue: (time_sec, frame_idx, grid18x12)
        self._infer_queue = deque(maxlen=8)
        self._infer_task = asyncio.ensure_future(self._inference_loop())

        # Defer-close mechanism so pred rows are written even if Stop happens quickly
        self._pending_close = False

        # Track last applied settings (avoid UI fighting user typing)
        self._last_settings_sensor_root = None
        self._last_settings_out_dir = None
        self._last_settings_base_name = None

        # Apply settings once at startup (script may have set them already)
        self._apply_settings(initial=True)

        # PhysX subscription
        self._physx = _physx.get_physx_interface()
        self._physx_sub = None
        try:
            self._physx_sub = self._physx.subscribe_physics_step_events(self._on_physics_step)
            print("[INFO] Subscribed to PhysX physics-step events.")
        except Exception as e:
            print("[ERROR] Failed to subscribe to PhysX physics-step events:", e)

        # Poll loop for selection + UI updates + stop detection
        self._task = asyncio.ensure_future(self._poll_loop())

        # Window menu entry
        self._menu_item = None
        try:
            import omni.kit.menu.utils as menu_utils
            self._menu_item = menu_utils.MenuItemDescription(
                name="CoRo Tactile Sensor",
                onclick_fn=self.show_window,
            )
            menu_utils.add_menu_items([self._menu_item], "Window")
            print("[INFO] Window menu item added: Window -> CoRo Tactile Sensor")
        except Exception as e:
            print("[WARN] Failed to add Window menu item:", e)

        print("[INFO] Extension started")

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
        self._model = None
        self._infer_fn = None
        self._clear_deformable_cache()

        print("[INFO] Extension shutdown complete.")

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
        prefix = getattr(self, "_settings_prefix", "/exts/coro_tactile")

        # sensor_root can be set by script; when set, lock picking
        sensor_root = s.get(f"{prefix}/sensor_root")
        sensor_root = str(sensor_root) if sensor_root else None
        if sensor_root and sensor_root != self._last_settings_sensor_root:
            self._last_settings_sensor_root = sensor_root
            self._active_sensor_root_path = sensor_root
            self._awaiting_new_sensor_pick = False
            self._clear_deformable_cache()

        # output_dir + base_name only when stopped & not logging (unless initial)
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

        # update only labels/previews (do NOT write into StringFields here)
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

    def _compute_output_paths(self):
        base = self._sanitize_base_name(self._base_name)
        out_dir = Path(self._output_dir).expanduser()
        dz_path = out_dir / f"{base}{SUFFIX_DZ}"
        pred_path = out_dir / f"{base}{SUFFIX_PRED}"
        return dz_path, pred_path

    # =========================================================
    # CSV: NODES
    # =========================================================
    def _load_nodes_csv(self):
        csv_path = EXT_ROOT / "data" / CSV_NAME
        if not csv_path.exists():
            print("[WARN] Nodes CSV not found:", csv_path)
            return []
        nodes = []
        with csv_path.open("r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                cell = str(row[0]).strip()
                if not cell or cell.lower() in {"node", "node_id", "nodeid", "id"}:
                    continue
                try:
                    nodes.append(int(cell))
                except Exception:
                    pass
        print(f"[INFO] Loaded {len(nodes)} node IDs")
        return nodes

    # =========================================================
    # LOAD: MAX VALUE
    # =========================================================
    def _load_max_value(self):
        print("[INFO] Max-value path:", MAXVAL_PATH)
        if not MAXVAL_PATH.exists():
            print("[WARN] Maximum_value_network.npy not found. Normalization disabled.")
            self._max_val = None
            return
        try:
            mv = np.load(str(MAXVAL_PATH))
            mv = np.asarray(mv, dtype=np.float32)
            self._max_val = float(mv) if mv.shape == () else float(np.max(mv))
            print("[INFO] Loaded max value:", self._max_val)
        except Exception as e:
            print("[WARN] Failed to load max value:", e)
            self._max_val = None

    # =========================================================
    # LOAD: SAVEDMODEL SIGNATURE
    # =========================================================
    def _load_savedmodel(self):
        mp = Path(SAVEDMODEL_DIR)
        print("[INFO] Model path:", mp)
        if not (mp.is_dir() and (mp / "saved_model.pb").exists()):
            print("[ERROR] SavedModel missing:", mp)
            self._infer_fn = None
            return

        try:
            sm = tf.saved_model.load(str(mp))
            sigs = getattr(sm, "signatures", None)

            if isinstance(sigs, Mapping) and len(sigs) > 0:
                keys = list(sigs.keys())
                print("[INFO] SavedModel signature keys:", keys)

                fn = sigs.get("serving_default", sigs[keys[0]])
                kw = fn.structured_input_signature[1]
                if isinstance(kw, dict) and len(kw) == 1:
                    self._infer_input_name = next(iter(kw.keys()))
                    print(f"[INFO] Signature input keyword: '{self._infer_input_name}'")
                else:
                    self._infer_input_name = None

                self._model = sm
                self._infer_fn = fn
                print("[INFO] Inference function ready.")
                return

            if callable(sm):
                self._model = sm
                self._infer_fn = sm
                self._infer_input_name = None
                print("[INFO] Loaded callable SavedModel (no signatures).")
                return

            print("[ERROR] SavedModel loaded but not callable and no signatures.")
            self._infer_fn = None
        except Exception as e:
            print("[ERROR] Failed to load SavedModel:", e)
            self._infer_fn = None

    # =========================================================
    # LOGGING DATA: START/STOP/FLUSH
    # =========================================================
    def _start_logging(self):
        if self._logging_active:
            return

        dz_path, pred_path = self._compute_output_paths()
        self._dz_path = dz_path
        self._pred_path = pred_path

        try:
            Path(self._output_dir).expanduser().mkdir(parents=True, exist_ok=True)

            self._dz_fh = self._dz_path.open("a", newline="")
            self._dz_writer = csv.writer(self._dz_fh)
            if self._dz_path.stat().st_size == 0:
                self._dz_writer.writerow(["time_sec", "frame"] + [f"dz_{i}" for i in range(EXPECTED_SIZE)])
                self._dz_fh.flush()

            self._pred_fh = self._pred_path.open("a", newline="")
            self._pred_writer = csv.writer(self._pred_fh)
            if self._pred_path.stat().st_size == 0:
                self._pred_writer.writerow(["time_sec", "frame"] + [f"pred_{i}" for i in range(28)])
                self._pred_fh.flush()

            self._logging_active = True
            self._since_flush = 0
            self._pending_close = False
            print(f"[LOG] Logging started -> {self._dz_path}")
            print(f"[LOG] Logging started -> {self._pred_path}")
        except Exception as e:
            print("[ERROR] Could not start logging:", e)
            self._close_files_now()

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

        self._dz_fh = None
        self._dz_writer = None
        self._pred_fh = None
        self._pred_writer = None
        self._logging_active = False
        self._since_flush = 0

    def _stop_logging(self):
        if self._infer_queue:
            self._pending_close = True
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
            except Exception:
                pass
            self._since_flush = 0

    # =========================================================
    # DEFORMABLE PATH 
    # =========================================================
    def _resolve_deform_path(self, sensor_root: str):
        stage = self._ctx.get_stage()

        deform_path = f"{sensor_root}/Soft/Deformable"
        prim = stage.GetPrimAtPath(deform_path)
        if prim and prim.IsValid():
            return deform_path

        soft = stage.GetPrimAtPath(f"{sensor_root}/Soft")
        if soft and soft.IsValid():
            for child in soft.GetChildren():
                if child.GetName() == "Deformable":
                    return child.GetPath().pathString
        return None

    def _clear_deformable_cache(self):
        self._cached_deform_container_path = None
        self._cached_deform_mesh_path = None
        self._cached_pos_attr = None
        self._cached_rest_attr = None
        self._cached_vel_attr = None

    def _find_first_mesh_under(self, container_path: str):
        stage = self._ctx.get_stage()
        root_prim = stage.GetPrimAtPath(container_path)
        if not root_prim or not root_prim.IsValid():
            return None
        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdGeom.Mesh):
                return prim
        return None

    def _ensure_deformable_cached(self, deform_container_path: str):
        if deform_container_path != self._cached_deform_container_path:
            self._clear_deformable_cache()
            self._cached_deform_container_path = deform_container_path

        if self._cached_pos_attr is not None and self._cached_pos_attr.IsValid():
            return True

        stage = self._ctx.get_stage()
        prim_at_path = stage.GetPrimAtPath(deform_container_path)
        if not prim_at_path or not prim_at_path.IsValid():
            return False

        if prim_at_path.IsA(UsdGeom.Mesh):
            mesh_prim = prim_at_path
        else:
            mesh_prim = self._find_first_mesh_under(deform_container_path)

        if mesh_prim is None:
            return False

        self._cached_deform_mesh_path = mesh_prim.GetPath().pathString
        pos_attr = mesh_prim.GetAttribute("physxDeformable:simulationPoints")
        rest_attr = mesh_prim.GetAttribute("physxDeformable:simulationRestPoints")
        vel_attr = mesh_prim.GetAttribute("physxDeformable:simulationVelocities")

        if not (pos_attr and pos_attr.IsValid()):
            return False

        self._cached_pos_attr = pos_attr
        self._cached_rest_attr = rest_attr if rest_attr and rest_attr.IsValid() else None
        self._cached_vel_attr = vel_attr if vel_attr and vel_attr.IsValid() else None
        return True

    def _read_deformable_arrays(self):
        tc = Usd.TimeCode.Default()
        sim_pts = self._cached_pos_attr.Get(tc) if self._cached_pos_attr else None
        rest_pts = self._cached_rest_attr.Get(tc) if self._cached_rest_attr else None
        return sim_pts, rest_pts

    # =========================================================
    # DZ COMPUTE + GRID
    # =========================================================
    def _valid_node_indices(self, points):
        if points is None:
            return []
        n = len(points)
        return [i for i in self._nodes_ids if isinstance(i, int) and 0 <= i < n]

    def _z_from_points(self, points, indices):
        return [float(points[i][2]) for i in indices]

    def _compute_dz(self, z_sim, z_rest):
        m = min(len(z_sim), len(z_rest))
        return [float(z_sim[i] - z_rest[i]) for i in range(m)]

    def _dz_to_grid(self, dz_list):
        arr = np.asarray(dz_list, dtype=np.float32)
        if arr.size != EXPECTED_SIZE:
            return None
        return arr.reshape((ROWS, COLS))

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

    def _pred_to_heatmap_rgba(self, pred: np.ndarray, h: int = 7, w: int = 4) -> np.ndarray:
        pred = np.asarray(pred, dtype=np.float32).reshape(-1)
        if pred.size < 28:
            pred = np.concatenate([pred, np.full((28 - pred.size,), np.nan, dtype=np.float32)], axis=0)
        pred = pred[:28]
        grid = pred.reshape(h, w)

        finite = np.isfinite(grid)
        if np.any(finite):
            vmin = float(np.nanmin(grid))
            vmax = float(np.nanmax(grid))
            if abs(vmax - vmin) < 1e-12:
                norm = np.zeros_like(grid, dtype=np.float32)
            else:
                norm = (grid - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(grid, dtype=np.float32)

        rgba = self._jet_rgba(norm)

        # NaNs → gray
        if np.any(~finite):
            rgba[~finite] = np.array([80, 80, 80, 255], dtype=np.uint8)

        return rgba

    def _write_heatmap_png(self, rgba_small: np.ndarray, out_path: str, out_w: int = 420, out_h: int = 240) -> bool:
        if Image is None:
            self._dbg_once("no_pillow", "[WARN] Pillow not available; cannot write heatmap PNG.")
            return False
        try:
            img = Image.fromarray(rgba_small, mode="RGBA")
            img = img.resize((out_w, out_h), resample=Image.NEAREST)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path, format="PNG")
            return True
        except Exception as e:
            self._dbg_once("png_write_err", f"[WARN] Failed to write heatmap PNG: {e}")
            return False

    def _update_heatmap_if_needed(self):
        if getattr(self, "_heatmap_img", None) is None:
            return
        if self._last_pred is None or self._last_pred_frame is None:
            return

        # checkbox toggle
        show = False
        if getattr(self, "_show_heatmap", None) is not None:
            try:
                show = bool(self._show_heatmap.model.get_value_as_bool())
            except Exception:
                show = False
        if not show:
            return

        # throttle
        now = float(self._time_sec())
        if (now - float(getattr(self, "_last_heatmap_update_t", -1e9))) < HEATMAP_MIN_DT:
            return

        rgba = self._pred_to_heatmap_rgba(self._last_pred, h=7, w=4)

        # alternate filenames to force refresh
        self._heatmap_png_toggle = not getattr(self, "_heatmap_png_toggle", False)
        out_path = self._heatmap_png_a if self._heatmap_png_toggle else self._heatmap_png_b

        ok = self._write_heatmap_png(rgba, str(out_path), out_w=300, out_h=600)
        if not ok:
            return

        try:
            # IMPORTANT: use a real file URI
            self._heatmap_img.source_url = out_path.as_uri()
            self._last_heatmap_update_t = now
        except Exception as e:
            self._dbg_once("heatmap_set_url", f"[WARN] Failed to set heatmap image url: {e}")


    # =========================================================
    # INFERENCE
    # =========================================================
    def _predict_from_grid(self, grid_18x12):
        if self._infer_fn is None:
            self._dbg_once("no_model", "[DBG] Inference disabled: SavedModel not loaded.")
            return None

        grid = np.asarray(grid_18x12, dtype=np.float32)
        if self._max_val is not None and abs(float(self._max_val)) > 1e-12:
            grid = grid / float(self._max_val)

        x = grid[None, :, :, None]  # (1,18,12,1)
        x_tf = tf.convert_to_tensor(x, dtype=tf.float32)

        if self._infer_input_name is not None:
            out = self._infer_fn(**{self._infer_input_name: x_tf})
        else:
            out = self._infer_fn(x_tf)

        if isinstance(out, dict) and len(out) > 0:
            y = next(iter(out.values()))
            return np.asarray(y.numpy()).reshape(-1)

        return np.asarray(out.numpy()).reshape(-1)

    async def _inference_loop(self):
        while getattr(self, "_running", False):
            try:
                if self._infer_queue:
                    t, frame_idx, grid = self._infer_queue.pop()
                    self._infer_queue.clear()

                    pred = self._predict_from_grid(grid)
                    if pred is None:
                        await asyncio.sleep(0.001)
                        continue

                    self._last_pred = pred
                    self._last_pred_frame = frame_idx
                    self._dbg_once("pred_ok", f"[DBG] Predictions are running (example pred0={float(pred[0]):.4f})")


                    if self._logging_active and self._pred_writer is not None:
                        pred28 = np.asarray(pred, dtype=np.float32).reshape(-1).tolist()
                        if len(pred28) < 28:
                            pred28 = pred28 + [float("nan")] * (28 - len(pred28))
                        else:
                            pred28 = pred28[:28]
                        self._pred_writer.writerow([f"{t:.6f}", int(frame_idx)] + [float(v) for v in pred28])
                        self._flush_if_needed()

                    if self._pending_close and not self._infer_queue:
                        self._pending_close = False
                        self._close_files_now()

            except Exception as e:
                self._dbg_once("infer_exc", f"[WARN] Inference loop error: {e}")

            await asyncio.sleep(0.001)

    # =========================================================
    # PHYSX STEP CONFIGURATION
    # =========================================================
    def _on_physics_step(self, dt: float):
        if not self._is_playing():
            return

        if not self._logging_active:
            self._start_logging()

        t = self._time_sec()
        frame_idx = self._frame_index(t)

        if self._last_processed_frame is not None and frame_idx == self._last_processed_frame:
            return
        self._last_processed_frame = frame_idx

        if not self._active_sensor_root_path:
            self._dbg_once("no_sensor", "[DBG] No sensor_root set. Set /exts/coro_tactile/sensor_root or pick in UI.")
            return

        deform_path = self._resolve_deform_path(self._active_sensor_root_path)
        if not deform_path:
            self._dbg_once("no_deform_path", f"[DBG] No deformable found at {self._active_sensor_root_path}/Soft/Deformable")
            return
        if not self._ensure_deformable_cached(deform_path):
            self._dbg_once("cache_fail", f"[DBG] Deformable cache failed under {deform_path}")
            return

        sim_pts, rest_pts = self._read_deformable_arrays()
        if sim_pts is None or rest_pts is None:
            self._dbg_once("pts_none", f"[DBG] sim/rest points missing on {self._cached_deform_mesh_path}")
            return

        idxs = self._valid_node_indices(sim_pts)
        if len(idxs) != EXPECTED_SIZE:
            self._dbg_once("idx_mismatch", f"[DBG] Node ids mismatch: {len(idxs)} != {EXPECTED_SIZE}")
            return

        z_sim = self._z_from_points(sim_pts, idxs)
        z_rest = self._z_from_points(rest_pts, idxs)
        dz = self._compute_dz(z_sim, z_rest)

        if len(dz) != EXPECTED_SIZE:
            self._dbg_once("dz_mismatch", f"[DBG] dz length {len(dz)} != {EXPECTED_SIZE}")
            return

        grid = self._dz_to_grid(dz)
        if grid is None:
            self._dbg_once("grid_none", "[DBG] grid reshape failed")
            return

        self._last_dz = dz
        self._last_time = t
        self._last_frame = frame_idx

        if self._dz_writer is not None:
            self._dz_writer.writerow([f"{t:.6f}", int(frame_idx)] + [float(v) for v in dz])
            self._flush_if_needed()

        self._infer_queue.append((t, frame_idx, grid))

    # =========================================================
    # UI
    # =========================================================

    def show_window(self, *_):
        if self._panel is None:
            self._panel = CoRoPanel(self)
        self._panel.show()

        # map panel widgets to your existing names (so the rest of your code still works)
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


    def _files_preview_text(self):
        dz_path, pred_path = self._compute_output_paths()
        return f"Files to generate:  {dz_path.name} and {pred_path.name}"

    def _can_edit_output(self) -> bool:
        return (not self._is_playing()) and (not self._logging_active)

    def _apply_output_dir(self):
        if not self._can_edit_output():
            print("[WARN] Stop simulation + click 'New run' before changing directory.")
            return

        try:
            raw = self._out_dir_field.model.get_value_as_string()
        except Exception:
            raw = ""

        raw = (raw or "").strip()
        if not raw:
            print("[WARN] Output directory is empty.")
            return

        try:
            p = Path(raw).expanduser()
            p.mkdir(parents=True, exist_ok=True)
            self._output_dir = p
        except Exception as e:
            print("[WARN] Invalid directory path:", raw, "|", e)
            return

        self._settings.set(f"{self._settings_prefix}/output_dir", str(self._output_dir))
        self._last_settings_out_dir = str(self._output_dir)

        if hasattr(self, "_out_dir_label") and self._out_dir_label is not None:
            self._out_dir_label.text = f"Directory: {self._output_dir}"
        if hasattr(self, "_files_preview") and self._files_preview is not None:
            self._files_preview.text = self._files_preview_text()

        print("[INFO] Output directory set to:", self._output_dir)

    def _apply_base_name(self):
        if not self._can_edit_output():
            print("[WARN] Stop simulation + click 'New run' before changing base name.")
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

        print("[INFO] Base name set to:", self._base_name)

    def _new_run(self):
        self._stop_logging()
        self._last_processed_frame = None
        self._reset_heatmap_state()   
        print("[INFO] New run ready. You can change name/directory now.")

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
        self._last_heatmap_frame = None

    # =========================================================
    # POLL LOOP 
    # =========================================================
    async def _poll_loop(self):
        while getattr(self, "_running", False):
            try:
                playing = self._is_playing()

                # Detect transition from stopped -> playing (new run start)
                if (not self._was_playing) and playing:
                    self._reset_heatmap_state()


                # pull updated settings (script can set after startup)
                self._apply_settings()

                # Detect transition from playing -> stopped
                if self._was_playing and (not playing):
                    if self._logging_active:
                        print("[INFO] Timeline stopped -> closing CSV files (may defer until inference drains).")
                        self._stop_logging()
                    self._last_processed_frame = None
                    self._reset_heatmap_state()   # <- add this

                self._was_playing = playing

                if self._window is not None and self._window.visible:
                    # sensor picking
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
                        deform_path = self._resolve_deform_path(self._active_sensor_root_path)
                        if deform_path and self._ensure_deformable_cached(deform_path):
                            if hasattr(self, "_def_mesh") and self._def_mesh is not None:
                                self._def_mesh.text = f"Mesh: {self._cached_deform_mesh_path}"
                        else:
                            if hasattr(self, "_def_mesh") and self._def_mesh is not None:
                                self._def_mesh.text = "Mesh: (not found)"

                    if self._last_dz is not None:
                        if hasattr(self, "_def_dz_count") and self._def_dz_count is not None:
                            self._def_dz_count.text = (
                                f"dZ count: {len(self._last_dz)} | frame={self._last_frame} | t={self._last_time:.4f}s"
                            )
                        if hasattr(self, "_def_dz_preview") and self._def_dz_preview is not None:
                            self._def_dz_preview.text = f"dZ preview: {self._preview_list(self._last_dz, k=6)}"
                    else:
                        if hasattr(self, "_def_dz_count") and self._def_dz_count is not None:
                            self._def_dz_count.text = "dZ count: (none)"
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

                    # Update heatmap (file-based)
                    self._update_heatmap_if_needed()

                    # Log label
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
                print("[WARN] UI poll error:", e)

            await asyncio.sleep(0.1)

    def _preview_list(self, arr, k=5):
        if arr is None or len(arr) == 0:
            return "(none)"
        vals = arr[:k]
        return "[" + ", ".join(f"{float(v):.6f}" for v in vals) + (", ...]" if len(arr) > k else "]")

