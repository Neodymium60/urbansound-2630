import os, time, sounddevice as sd, numpy as np, wave, datetime as dt

DATA_DIR = os.getenv("DATA_DIR","/data")
SITE_ID  = os.getenv("SITE_ID","site")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE","44100"))
THRESH_DB   = float(os.getenv("NOISE_DB_THRESHOLD","65"))
CLIP_SECONDS= float(os.getenv("CLIP_SECONDS","3"))
BLOCKSIZE   = int(os.getenv("BLOCKSIZE","4096"))
LATENCY     = os.getenv("LATENCY","high")
MIN_GAP     = float(os.getenv("MIN_GAP_SECONDS","1.5"))
MIC_SPEC    = os.getenv("MIC_DEVICE","")

def resolve_device(spec):
    if spec.strip().isdigit():
        return int(spec)
    s = spec.lower()
    for i,d in enumerate(sd.query_devices()):
        if d["max_input_channels"]>0 and s in d["name"].lower():
            return i
    raise ValueError(f"No input device matching '{spec}'")

def rms_dbfs(x):
    x = x.astype(np.float32)/32768.0
    rms = np.sqrt(np.mean(x**2) + 1e-12)
    return 20*np.log10(rms + 1e-12)

def write_wav(path, sr, x):
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(x.tobytes())

def main():
    device = resolve_device(MIC_SPEC) if MIC_SPEC else None
    print(f"[recorder] cfg device={MIC_SPEC or device}, rate={SAMPLE_RATE}, "
          f"threshold={THRESH_DB}, clip={CLIP_SECONDS}s, block={BLOCKSIZE}, lat={LATENCY}", flush=True)

    clip = []
    last_write = 0
    triggered = False

    def callback(indata, frames, time_info, status):
        nonlocal clip, triggered, last_write
        if status.input_overflow:
            print("input overflow", flush=True)
        mono = indata[:,0].copy()
        db = rms_dbfs(mono)
        if db* -1 < 0:  # quiet guard (db is negative dBFS)
            pass
        if db >= - (100-THRESH_DB):  # map NOISE_DB_THRESHOLD ~65 to dBFS gate
            clip.append(mono.copy())
            if not triggered:
                triggered = True
        elif triggered and (time.time()-last_write) >= MIN_GAP:
            # finalize
            y = np.concatenate(clip) if clip else np.zeros(1,dtype=np.float32)
            clip = []; triggered=False
            y = (np.clip(y, -1, 1)*32767).astype(np.int16)
            if len(y) < int(CLIP_SECONDS*SAMPLE_RATE):
                pad = np.zeros(int(CLIP_SECONDS*SAMPLE_RATE)-len(y), np.int16)
                y = np.concatenate([y, pad])
            ts = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            path = os.path.join(DATA_DIR, f"{ts}.wav")
            write_wav(path, SAMPLE_RATE, y)
            last_write = time.time()
            print(f"[recorder] wrote {path} ({CLIP_SECONDS:.1f}s)", flush=True)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=BLOCKSIZE, latency=LATENCY, device=device, callback=callback):
        while True:
            time.sleep(0.25)

if __name__ == "__main__":
    main()

