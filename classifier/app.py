#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, time, wave, contextlib, json, logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple, List

import numpy as np

# Assuming tflite_runtime is installed. If not, replace with tensorflow.lite
from tflite_runtime.interpreter import Interpreter, load_delegate

# -----------------------
# Config (env)
# -----------------------
DATA_DIR        = os.getenv("DATA_DIR", "/data")
MODEL_PATH      = os.getenv("MODEL_PATH", "/app/model.tflite")
LABELS_PATH     = os.getenv("LABELS_PATH", "/app/labels.txt")
USE_TPU         = os.getenv("USE_TPU", "false").lower() == "true"
SITE_ID         = os.getenv("SITE_ID", "unknown")
CAL_DB_OFFSET   = float(os.getenv("CAL_DB_OFFSET", "0"))  # dB added to dBFS to approximate dBA after field calibration
CSV_PATH        = os.path.join(DATA_DIR, "labels.csv")
POLL_SEC        = float(os.getenv("POLL_SEC", "1.0"))     # scan interval for new WAVs
MIN_AGE_SEC     = float(os.getenv("MIN_AGE_SEC", "0.2"))  # wait so writer closes file first
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO").upper()

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="[classifier] %(message)s",
)
log = logging.getLogger("urbannoise-classifier")

# -----------------------
# CSV header
# -----------------------
CSV_HEADER = [
    "timestamp", "filename", "label", "confidence", "site_id",
    "dbfs_rms", "dbfs_peak", "spl_est_dbA"
]

def ensure_csv_header(path: str = CSV_PATH) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

def already_indexed() -> set:
    """Return filenames already present in CSV (to avoid duplicates)."""
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        return set()
    seen = set()
    with open(CSV_PATH, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            fn = row.get("filename")
            if fn:
                seen.add(fn)
    return seen

# -----------------------
# Audio utils
# -----------------------
def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """Read mono/multi-channel PCM WAV. Returns (float32 mono in [-1,1], sr)."""
    with contextlib.closing(wave.open(path, "rb")) as wf:
        nc = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        n  = wf.getnframes()
        frames = wf.readframes(n)

    if sw == 1:
        raw = np.frombuffer(frames, dtype=np.uint8).astype(np.int16)
        raw = (raw - 128) * 256
        peak_int = 32767.0
    elif sw == 2:
        raw = np.frombuffer(frames, dtype=np.int16)
        peak_int = 32767.0
    else:
        # 24/32-bit -> read as int32
        raw = np.frombuffer(frames, dtype=np.int32)
        peak_int = 2147483647.0

    if nc > 1:
        raw = raw.reshape(-1, nc).mean(axis=1)

    x = (raw.astype(np.float32) / peak_int).clip(-1.0, 1.0)
    return x, sr

def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Lightweight linear resampler."""
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32, copy=False)
    duration = x.size / float(sr_in)
    n_out = int(round(duration * sr_out))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    xp = np.linspace(0.0, 1.0, num=x.size, endpoint=False, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float32)
    return np.interp(x_new, xp, x.astype(np.float32)).astype(np.float32)

def compute_levels_from_float(x: np.ndarray) -> Tuple[float, float]:
    """Return (dbfs_rms, dbfs_peak) for float32 x in [-1,1]."""
    if x.size == 0:
        return float("nan"), float("nan")
    rms  = np.sqrt(np.mean(x * x) + 1e-12)
    peak = np.max(np.abs(x)) + 1e-12
    return float(20.0 * np.log10(rms)), float(20.0 * np.log10(peak))

# -----------------------
# TFLite wrapper
# -----------------------
def softmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float32)
    e = np.exp(z - np.max(z))
    s = e / np.maximum(np.sum(e), 1e-9)
    return s

class LiteModel:
    def __init__(self, model_path: str, use_tpu: bool, labels_path: Optional[str]):
        self.loaded = False
        self.labels = None
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, "r", encoding="utf-8") as f:
                self.labels = [line.strip() for line in f.readlines()]
        
        delegates = [load_delegate("libedgetpu.so.1")] if use_tpu else None
        
        self.interp = Interpreter(model_path=model_path, experimental_delegates=delegates)
        self.interp.allocate_tensors()
        self.in_det  = self.interp.get_input_details()[0]
        self.out_det = self.interp.get_output_details()[0]

        # --- FIX: canonicalize to rank-2 [1, 15600] input -----------------
        try:
            tgt_shape = [1, 15600] # <-- CHANGED
            in_idx = self.in_det["index"]
            # If input shape is not exactly [1, 15600], resize it.
            if list(self.in_det["shape"]) != tgt_shape: # <-- CHANGED
                self.interp.resize_tensor_input(in_idx, tgt_shape) # <-- CHANGED
                self.interp.allocate_tensors()
                self.in_det  = self.interp.get_input_details()[0]
                self.out_det = self.interp.get_output_details()[0]
                log.info(f"normalized input shape to {list(self.in_det['shape'])}")
        except Exception as e:
            log.warning(f"could not normalize input shape: {e}")
        # ----------------------------------------------------------------

        self.loaded = True
        log.info(f"loaded model={model_path} tpu={use_tpu} "
                 f"input={self.in_det['shape']} dtype={self.in_det['dtype']} "
                 f"output={self.out_det['shape']}")

    def _prepare_input(self, wav_path: str):
        x, sr = read_wav(wav_path)
        dbfs_rms, dbfs_peak = compute_levels_from_float(x)
        x16 = resample_linear(x, sr, 16000) if sr != 16000 else x.astype(np.float32, copy=False)

        # --- NEW: always return 1-D vector of length 15600 ---------------
        tgt_len = 15600
        if x16.size < tgt_len:
            x_pad = np.zeros(tgt_len, dtype=np.float32)
            x_pad[:x16.size] = x16
            x16 = x_pad
        else:
            x16 = x16[:tgt_len].astype(np.float32, copy=False)
        x_in = x16  # strictly 1-D for now
        # -----------------------------------------------------------------

        # Quantization handling (unchanged)
        in_dtype = self.in_det["dtype"] if self.in_det else np.float32
        q = self.in_det.get("quantization", None) if self.in_det else None
        if q and isinstance(q, (list, tuple)) and len(q) == 2 and q[0] != 0:
            scale, zero = q
            if np.issubdtype(in_dtype, np.integer):
                if in_dtype == np.int8:
                    x_in = (x_in / scale + zero).astype(np.int8)
                elif in_dtype == np.uint8:
                    x_in = (x_in / scale + zero).astype(np.uint8)
                else:
                    x_in = (x_in / scale + zero).astype(in_dtype)
        return x_in, dbfs_rms, dbfs_peak

    def infer(self, wav_path: str):
        x_in, dbfs_rms, dbfs_peak = self._prepare_input(wav_path)
        if x_in is None:
            return "unknown", 0.5, dbfs_rms, dbfs_peak

        try:
            # --- FIX: Reshape the input to 2D tensor [1, 15600] ---
            x_in_reshaped = x_in.reshape(1, -1) # <-- ADDED
            self.interp.set_tensor(self.in_det["index"], x_in_reshaped) # <-- CHANGED
            self.interp.invoke()
            out = np.squeeze(self.interp.get_tensor(self.out_det["index"]).astype(np.float32))
            
            # Post-processing
            probs = softmax(out) if (np.any(out < 0) or np.sum(out) <= 0 or np.sum(out) > 1.01) else out
            idx = int(np.argmax(probs))
            conf = float(np.max(probs))
            lbl = self.labels[idx] if self.labels and 0 <= idx < len(self.labels) else f"class_{idx}"
            return lbl, conf, dbfs_rms, dbfs_peak

        except Exception as e:
            # One more retry without delegates (rare, but helps some builds)
            try:
                log.warning(f"inference failed once ({e}); retrying without delegates")
                from tflite_runtime.interpreter import Interpreter as _I
                i2 = _I(model_path=MODEL_PATH)
                i2.allocate_tensors()
                # --- FIX: Use 2D shape in retry ---
                i2.resize_tensor_input(i2.get_input_details()[0]["index"], [1, 15600]) # <-- CHANGED
                i2.allocate_tensors()
                # --- FIX: Reshape input in retry ---
                x_in_reshaped_retry = x_in.astype(np.float32).reshape(1, -1) # <-- ADDED
                i2.set_tensor(i2.get_input_details()[0]["index"], x_in_reshaped_retry) # <-- CHANGED
                i2.invoke()
                out = np.squeeze(i2.get_tensor(i2.get_output_details()[0]["index"]).astype(np.float32))
                probs = softmax(out) if (np.any(out < 0) or np.sum(out) <= 0 or np.sum(out) > 1.01) else out
                idx = int(np.argmax(probs)); conf = float(np.max(probs))
                lbl = self.labels[idx] if self.labels and 0 <= idx < len(self.labels) else f"class_{idx}"
                # Note: compute_levels_from_float still uses the original 1D x_in
                return lbl, conf, *compute_levels_from_float(x_in)
            except Exception as e2:
                log.warning(f"inference failed on {Path(wav_path).name}: {e2}")
                return "unknown", 0.5, dbfs_rms, dbfs_peak


# -----------------------
# CSV writing
# -----------------------
def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def write_row(ts_iso: str, fname: str, label: str, conf: float, dbfs_rms: float, dbfs_peak: float) -> None:
    spl_est = dbfs_rms + CAL_DB_OFFSET if np.isfinite(dbfs_rms) else float("nan")
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            ts_iso,
            fname,
            label,                                  # csv.writer handles commas/quotes
            round(float(conf), 3),
            SITE_ID,
            None if not np.isfinite(dbfs_rms) else round(dbfs_rms, 1),
            None if not np.isfinite(dbfs_peak) else round(dbfs_peak, 1),
            None if not np.isfinite(spl_est) else round(spl_est, 1),
        ])

def should_pick(file: Path, seen: set) -> bool:
    if not file.is_file() or file.suffix.lower() != ".wav":
        return False
    if file.name in seen:
        return False
    age = time.time() - file.stat().st_mtime
    return age >= MIN_AGE_SEC

# -----------------------
# Main loop
# -----------------------
def main() -> None:
    log.info(json.dumps({
        "msg": "starting",
        "data_dir": DATA_DIR,
        "model": MODEL_PATH,
        "labels": LABELS_PATH if os.path.exists(LABELS_PATH) else None,
        "use_tpu": USE_TPU,
        "site_id": SITE_ID,
        "cal_db_offset": CAL_DB_OFFSET
    }))

    ensure_csv_header()
    seen = already_indexed()
    mdl = LiteModel(MODEL_PATH, USE_TPU, LABELS_PATH)

    # Process backlog first
    for p in sorted(Path(DATA_DIR).glob("*.wav")):
        if should_pick(p, seen):
            ts = iso_now()
            label, conf, dbfs_rms, dbfs_peak = mdl.infer(str(p))
            write_row(ts, p.name, label, conf, dbfs_rms, dbfs_peak)
            seen.add(p.name)
            log.info(f"classified backlog {p.name} => {label} ({conf:.2f})")

    # Watch loop
    while True:
        try:
            for p in sorted(Path(DATA_DIR).glob("*.wav")):
                if should_pick(p, seen):
                    ts = iso_now()
                    label, conf, dbfs_rms, dbfs_peak = mdl.infer(str(p))
                    write_row(ts, p.name, label, conf, dbfs_rms, dbfs_peak)
                    seen.add(p.name)
                    log.info(f"classified {p.name} => {label} ({conf:.2f})")
        except Exception as e:
            log.warning(f"loop error: {e}")
        time.sleep(POLL_SEC)

if __name__ == "__main__":
    main()
