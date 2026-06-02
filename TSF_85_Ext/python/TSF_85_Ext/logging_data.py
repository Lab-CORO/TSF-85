import csv
from pathlib import Path


class DataLogger:
    def __init__(self, output_dir, base_name, expected_size):
        self.output_dir = Path(output_dir)
        self.base_name = base_name
        self.expected_size = expected_size

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

    # -------------------------------------------------
    # PATHS
    # -------------------------------------------------
    def _compute_paths(self):
        dz_path = self.output_dir / f"{self.base_name}_deformations.csv"
        pred_path = self.output_dir / f"{self.base_name}_tactile_maps.csv"
        mesh_path = self.output_dir / f"{self.base_name}_mesh_state.csv"
        return dz_path, pred_path, mesh_path

    # -------------------------------------------------
    # START
    # -------------------------------------------------
    def start(self):
        if self._logging_active:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        dz_path, pred_path, mesh_path = self._compute_paths()
        self._dz_path = dz_path
        self._pred_path = pred_path
        self._mesh_path = mesh_path

        try:
            # ---- DZ FILE ----
            self._dz_fh = dz_path.open("a", newline="")
            self._dz_writer = csv.writer(self._dz_fh)

            if dz_path.stat().st_size == 0:
                header = ["time_sec", "frame"] + [f"dz_{i}" for i in range(self.expected_size)]
                self._dz_writer.writerow(header)
                self._dz_fh.flush()

            # ---- PRED FILE ----
            self._pred_fh = pred_path.open("a", newline="")
            self._pred_writer = csv.writer(self._pred_fh)

            if pred_path.stat().st_size == 0:
                header = ["time_sec", "frame"] + [f"pred_{i}" for i in range(28)]
                self._pred_writer.writerow(header)
                self._pred_fh.flush()

            # ---- MESH STATE FILE ----
            # One row per frame, wide layout:
            #   time_sec, frame,
            #   case_x, case_y, case_z, case_qw, case_qx, case_qy, case_qz,
            #   node0_x, node0_y, node0_z, node0_rx, node0_ry, node0_rz,
            #   node1_x, ...
            # for all `expected_size` filtered nodes, in case-local frame.
            self._mesh_fh = mesh_path.open("a", newline="")
            self._mesh_writer = csv.writer(self._mesh_fh)

            if mesh_path.stat().st_size == 0:
                header = [
                    "time_sec", "frame",
                    "case_x", "case_y", "case_z",
                    "case_qw", "case_qx", "case_qy", "case_qz",
                ]
                for i in range(self.expected_size):
                    header += [
                        f"node{i}_x", f"node{i}_y", f"node{i}_z",
                        f"node{i}_rx", f"node{i}_ry", f"node{i}_rz",
                    ]
                self._mesh_writer.writerow(header)
                self._mesh_fh.flush()

            self._logging_active = True
            self._since_flush = 0

            print(f"[TSF-85][LOG] Logging started -> {dz_path}")
            print(f"[TSF-85][LOG] Logging started -> {pred_path}")
            print(f"[TSF-85][LOG] Logging started -> {mesh_path}")

        except Exception as e:
            print("[TSF-85][ERROR] Logger start failed:", e)
            self.stop()

    # -------------------------------------------------
    # WRITE DZ
    # -------------------------------------------------
    def log_dz(self, t, frame, dz):
        if not self._logging_active or self._dz_writer is None:
            return

        self._dz_writer.writerow([f"{t:.6f}", int(frame)] + [float(v) for v in dz])
        self._flush_if_needed()

    # -------------------------------------------------
    # WRITE PREDICTION
    # -------------------------------------------------
    def log_pred(self, t, frame, pred):
        if not self._logging_active or self._pred_writer is None:
            return

        pred = list(pred)

        if len(pred) < 28:
            pred = pred + [float("nan")] * (28 - len(pred))
        else:
            pred = pred[:28]

        self._pred_writer.writerow([f"{t:.6f}", int(frame)] + pred)
        self._flush_if_needed()

    # -------------------------------------------------
    # WRITE MESH STATE
    # -------------------------------------------------
    def log_mesh_state(self, t, frame, case_translate, case_quat, curr_f, rest_f):
        """Write one wide row with case pose + filtered node positions.

        case_translate : array-like length 3   (case_x, case_y, case_z) in world.
        case_quat      : array-like length 4   (qw, qx, qy, qz) of the case.
        curr_f         : np.ndarray shape (N, 3) current positions of filtered nodes,
                         in case-local frame.
        rest_f         : np.ndarray shape (N, 3) rest positions of filtered nodes,
                         in case-local frame.
        N must equal self.expected_size.
        """
        if not self._logging_active or self._mesh_writer is None:
            return

        row = [f"{t:.6f}", int(frame)]
        row += [float(case_translate[0]), float(case_translate[1]), float(case_translate[2])]
        row += [float(case_quat[0]), float(case_quat[1]),
                float(case_quat[2]), float(case_quat[3])]

        n = self.expected_size
        for i in range(n):
            if i < len(curr_f):
                cx, cy, cz = float(curr_f[i][0]), float(curr_f[i][1]), float(curr_f[i][2])
            else:
                cx = cy = cz = float("nan")
            if i < len(rest_f):
                rx, ry, rz = float(rest_f[i][0]), float(rest_f[i][1]), float(rest_f[i][2])
            else:
                rx = ry = rz = float("nan")
            row += [cx, cy, cz, rx, ry, rz]

        self._mesh_writer.writerow(row)
        self._flush_if_needed()

    # -------------------------------------------------
    # FLUSH
    # -------------------------------------------------
    def _flush_if_needed(self):
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

            self._since_flush = 0

    # -------------------------------------------------
    # STOP
    # -------------------------------------------------
    def stop(self):
        for fh in (self._dz_fh, self._pred_fh, self._mesh_fh):
            try:
                if fh:
                    fh.flush()
                    fh.close()
            except Exception:
                pass

        self._dz_fh = None
        self._pred_fh = None
        self._mesh_fh = None
        self._dz_writer = None
        self._pred_writer = None
        self._mesh_writer = None
        self._logging_active = False
        self._since_flush = 0

        print("[TSF-85][LOG] Logging stopped")

    # -------------------------------------------------
    def is_active(self):
        return self._logging_active
