import csv
from pathlib import Path


class DataLogger:
    def __init__(self, output_dir, base_name, expected_size):
        self.output_dir = Path(output_dir)
        self.base_name = base_name
        self.expected_size = expected_size

        self._dz_path = None
        self._pred_path = None

        self._dz_fh = None
        self._dz_writer = None

        self._pred_fh = None
        self._pred_writer = None

        self._logging_active = False
        self._flush_every = 50
        self._since_flush = 0

    # -------------------------------------------------
    # PATHS
    # -------------------------------------------------
    def _compute_paths(self):
        dz_path = self.output_dir / f"{self.base_name}_deformations.csv"
        pred_path = self.output_dir / f"{self.base_name}_tactile_maps.csv"
        return dz_path, pred_path

    # -------------------------------------------------
    # START
    # -------------------------------------------------
    def start(self):
        if self._logging_active:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        dz_path, pred_path = self._compute_paths()
        self._dz_path = dz_path
        self._pred_path = pred_path

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

            self._logging_active = True
            self._since_flush = 0

            print(f"[LOG] Logging started -> {dz_path}")
            print(f"[LOG] Logging started -> {pred_path}")

        except Exception as e:
            print("[ERROR] Logger start failed:", e)
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
            except Exception:
                pass

            self._since_flush = 0

    # -------------------------------------------------
    # STOP
    # -------------------------------------------------
    def stop(self):
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
        self._pred_fh = None
        self._dz_writer = None
        self._pred_writer = None
        self._logging_active = False
        self._since_flush = 0

        print("[LOG] Logging stopped")

    # -------------------------------------------------
    def is_active(self):
        return self._logging_active
