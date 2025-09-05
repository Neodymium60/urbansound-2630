#!/usr/bin/env python3
import os, csv, time, wave, contextlib, logging, json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List
import numpy as np

# -----------------------
# Config (env)
# -----------------------
DATA_DIR        = os.getenv("DATA_DIR", "/data")
MODEL_PATH      = os.getenv("MODEL_PATH", "/app/model.tflite")
LABELS_PATH     = os.getenv("LABELS_PATH", "/app/labels.txt")
USE_TPU         = os.getenv("USE_TPU", "false").lower() == "true"
SITE_ID         = os.getenv("SITE_ID", "unknown")
CAL_DB_OFFSET   = float(os.getenv("CAL_DB_OFFSET", "0"))  # dB to convert dBFS->approx SPL dBA after calibration
CSV_PATH        = os.path.join(DATA_DIR, "labels.csv")
POLL_SEC        = float(os.getenv("POLL_SEC", "1.0"))     # scan interval for new WAVs
MIN_AGE_SEC     = float(os.getenv("MIN_AGE_SEC", "0.2"))  # wait so writer closes file
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
# CSV schema
# -----------------------
CSV_HEADER = [
    "timestamp", "filename", "label", "confidence", "site_id",
    "dbfs_rms", "dbfs_peak", "spl_est_dbA"
]

def ensure_csv_header(path: str = CSV_PATH):
    """Create CSV with header if missing; fsync to reduce partial-write risk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)
            f.flush(); os.fsync(f.fileno())

def already_indexed() -> set:
    """
    Return set of filenames already present in CSV (avoid duplicates).
    Tolerate corrupt CSV (e.g., NULs or parse errors) by rotating it once.
    """
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        return set()
    try:
        seen = set()
        with open(CSV_PATH, "r", newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                fn = (row or {}).get("filename")
                if fn:
                    seen.add(fn)
        return seen
    except Exception as e:
        log.warning(f"labels.csv unreadable ({e}); rotating and starting fresh")
        try:
            os.replace(CSV_PATH, CSV_PATH + ".damaged")
        except Exception:
            pass
        ensure_csv_header()
        return set()

# -----------------------
# Audio & level utilities
# -----------------------
def read_wav(path: str):
    """Read mono or multi-channel PCM WAV. Returns (audio_float32[-1..1], sr)."""
    with contextlib.closing(wave.open(path, "rb")) as wf:
        nc = wf.getnchannels()
        sw = wf.getsampwidth()  # bytes/sample
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
        raw = np.frombuffer(frames, dtype=np.int32)
        peak_int = 2147483647.0

    if raw.size == 0:
        return np.zeros(0, dtype=np.float32), sr

    if nc > 1:
        try:
            raw = raw.reshape(-1, nc).mean(axis=1)
        except Exception:
            # shape mismatch => fall back to mono pick
            raw = raw[::nc]

    x = (raw.astype(np.float32) / peak_int)
    return x, sr

def compute_levels_from_float(x: np.ndarray):
    """Return (dbfs_rms, dbfs_peak) for float32 x in [-1,1]."""
    if x.size == 0:
        return float("nan"), float("nan")
    rms  = np.sqrt(np.mean(x * x) + 1e-12)
    peak = np.max(np.abs(x)) + 1e-12
    dbfs_rms  = 20.0 * np.log10(rms)
    dbfs_peak = 20.0 * np.log10(peak)
    return float(dbfs_rms), float(dbfs_peak)

def compute_levels(wav_path: str):
    x, _ = read_wav(wav_path)
    return compute_levels_from_float(x)

def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Very light linear resampler (sufficient for inference)."""
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32, copy=False)
    duration = x.size / float(sr_in)
    n_out = int(round(duration * sr_out))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    xp = np.linspace(0.0, 1.0, num=x.size, endpoint=False, dtype=np.float32)
    fp = x.astype(np.float32)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float32)
    return np.interp(x_new, xp, fp).astype(np.float32)

# -----------------------
# TFLite model wrapper
# -----------------------
def _target_len_from_shape(shape: List[int]) -> int:
    """
    Infer target waveform length from input tensor shape.
    Handles 1D [N], 2D [1,N] or [N,1], and 3D [1,N,1]. Defaults to 15600 (YAMNet).
    """
    shp = list(shape)
    if len(shp) == 1:
        return int(shp[0])
    if len(shp) == 2:
        return int(max(shp))
    if len(shp) == 3:
        return int(shp[1])
    return 15600

class LiteModel:
    def __init__(self, model_path: str, use_tpu: bool, labels_path: Optional[str]):
        self.interp = None
        self.input_details = None
        self.output_details = None
        self.labels = None
        self.loaded = False

        if labels_path and os.path.exists(labels_path):
            with open(labels_path, "r", encoding="utf-8", errors="ignore") as f:
                self.labels = [ln.rstrip("\n") for ln in f if ln.strip()]

        if not os.path.exists(model_path):
            log.info(f"model not found: {model_path} — will use fallback rule-based labels")
            return

        # Prefer tflite_runtime (small), fall back to TF if present
        Interpreter = None
        load_delegate = None
        try:
            from tflite_runtime.interpreter import Interpreter as TRInterpreter, load_delegate as TRLoad
            Interpreter, load_delegate = TRInterpreter, TRLoad
        except Exception:
            try:
                from tensorflow.lite.python.interpreter import Interpreter as TFInterpreter  # type: ignore
                Interpreter = TFInterpreter
            except Exception as e:
                log.warning(f"tflite not available ({e}); using fallback labels")
                return

        try:
            if use_tpu and load_delegate:
                delegates = []
                for name in ("libedgetpu.so.1", "libedgetpu.so.1.0", "edgetpu.dll"):
                    try:
                        delegates = [load_delegate(name)]
                        break
                    except Exception:
                        continue
                self.interp = Interpreter(model_path=model_path, experimental_delegates=delegates)
            else:
                self.interp = Interpreter(model_path=model_path)

            self.interp.allocate_tensors()
            self.input_details  = self.interp.get_input_details()
            self.output_details = self.interp.get_output_details()
            self.loaded = True
            log.info(f"loaded model={model_path} tpu={use_tpu} input={self.input_details[0]['shape']} dtype={self.input_details[0]['dtype']} output={self.output_details[0]['shape']}")
        except Exception as e:
            log.warning(f"failed to load tflite model ({e}); using fallback labels")

    def _prepare_input(self, wav_path: str):
        """Load WAV, resample to 16k, and shape to model input."""
        x, sr = read_wav(wav_path)
        # YAMNet expects ~0.975s @ 16kHz => 15600 samples
        x16 = resample_linear(x, sr, 16000).astype(np.float32)

        in_det  = self.input_details[0]
        in_shape = list(in_det["shape"])
        in_dtype = in_det["dtype"]
        target_len = _target_len_from_shape(in_shape)

        # pad/clip
        if x16.size < target_len:
            buf = np.zeros(target_len, dtype=np.float32)
            buf[:x16.size] = x16
            x16 = buf
        else:
            x16 = x16[:target_len]

        # quantization?
        q = in_det.get("quantization")
        if q and isinstance(q, (tuple, list)) and len(q) == 2 and q[0] not in (None, 0.0):
            scale, zero = q
            xq = np.round(x16 / scale + zero)
            if in_dtype == np.int8:
                xq = np.clip(xq, -128, 127).astype(np.int8)
            elif in_dtype == np.uint8:
                xq = np.clip(xq, 0, 255).astype(np.uint8)
            else:
                xq = xq.astype(in_dtype)
            x_in = xq
        else:
            x_in = x16.astype(np.float32)

        # match shape
        if len(in_shape) == 1:         # [N]
            return x_in
        elif len(in_shape) == 2:       # [1,N] or [N,1]
            if in_shape[0] == 1:
                return x_in[None, :]
            else:
                return x_in[:, None]
        elif len(in_shape) == 3:       # [1,N,1]
            return x_in[None, :, None]
        else:
            return x_in  # fallback best effort

    def infer(self, wav_path: str):
        """
        Returns (label:str, confidence:float) using the model if loaded,
        else a simple rule-based label from SPL.
        """
        # Always compute levels (needed for CSV & fallback)
        dbfs_rms, dbfs_peak = compute_levels(wav_path)
        spl_est = dbfs_rms + CAL_DB_OFFSET if np.isfinite(dbfs_rms) else float("nan")

        if not self.loaded:
            # SPL rule-of-thumb fallback
            if np.isfinite(spl_est):
                if spl_est < 45:
                    return "Silence", 0.50
                elif spl_est < 60:
                    return "Moderate", 0.60
                elif spl_est < 75:
                    return "Traffic", 0.65
                else:
                    return "Very loud", 0.70
            return "unknown", 0.50

        try:
            x_in = self._prepare_input(wav_path)
            self.interp.set_tensor(self.input_details[0]["index"], x_in)
            self.interp.invoke()

            out_det = self.output_details[0]
            out = self.interp.get_tensor(out_det["index"]).squeeze()

            # dequantize if necessary
            q = out_det.get("quantization")
            if q and isinstance(q, (tuple, list)) and len(q) == 2 and q[0] not in (None, 0.0):
                scale, zero = q
                out = (out.astype(np.float32) - zero) * scale

            # ensure probs
            probs = out.astype(np.float32)
            if probs.ndim == 0:
                probs = np.array([1.0 - float(probs), float(probs)], dtype=np.float32)
            if (probs < 0).any() or probs.sum() <= 0 or probs.sum() > 1.01:
                e = np.exp(probs - np.max(probs))
                probs = e / np.sum(e)

            idx = int(np.argmax(probs))
            conf = float(np.max(probs))
            if self.labels and 0 <= idx < len(self.labels):
                lbl = self.labels[idx]
            else:
                lbl = f"class_{idx}"
            return lbl, conf
        except Exception as e:
            log.warning(f"inference failed on {Path(wav_path).name}: {e}")
            return "unknown", 0.50

# -----------------------
# CSV writing
# -----------------------
def write_row(ts_iso: str, fname: str, label: str, conf: float, dbfs_rms: float, dbfs_peak: float):
    spl_est = dbfs_rms + CAL_DB_OFFSET if np.isfinite(dbfs_rms) else float("nan")
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            ts_iso,
            fname,
            label,
            round(float(conf), 3),
            SITE_ID,
            None if not np.isfinite(dbfs_rms) else round(dbfs_rms, 1),
            None if not np.isfinite(dbfs_peak) else round(dbfs_peak, 1),
            None if not np.isfinite(spl_est) else round(spl_est, 1),
        ])
        f.flush(); os.fsync(f.fileno())

# -----------------------
# Main loop
# -----------------------
def iso_now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def should_pick(file: Path, seen: set) -> bool:
    if not file.is_file():
        return False
    if file.suffix.lower() != ".wav":
        return False
    if file.name in seen:
        return False
    # ensure writer has closed the file
    age = time.time() - file.stat().st_mtime
    return age >= MIN_AGE_SEC

def main():
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

    # process backlog first
    for p in sorted(Path(DATA_DIR).glob("*.wav")):
        if should_pick(p, seen):
            ts = iso_now()
            dbfs_rms, dbfs_peak = compute_levels(str(p))
            label, conf = mdl.infer(str(p))
            write_row(ts, p.name, label, conf, dbfs_rms, dbfs_peak)
            seen.add(p.name)
            log.info(f"classified backlog {p.name} => {label} ({conf:.2f})")

    # watch loop
    while True:
        try:
            for p in sorted(Path(DATA_DIR).glob("*.wav")):
                if should_pick(p, seen):
                    ts = iso_now()
                    dbfs_rms, dbfs_peak = compute_levels(str(p))
                    label, conf = mdl.infer(str(p))
                    write_row(ts, p.name, label, conf, dbfs_rms, dbfs_peak)
                    seen.add(p.name)
                    log.info(f"classified {p.name} => {label} ({conf:.2f})")
        except Exception as e:
            log.warning(f"loop error: {e}")
        time.sleep(POLL_SEC)

if __name__ == "__main__":
    main()

