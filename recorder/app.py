import os, time, sounddevice as sd, numpy as np, wave, datetime as dt, queue, threading

# --- Configuration (from environment variables) ---
DATA_DIR = os.getenv("DATA_DIR", "/data")
SITE_ID = os.getenv("SITE_ID", "site")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "44100"))
THRESH_DB = float(os.getenv("NOISE_DB_THRESHOLD", "65"))
CLIP_SECONDS = float(os.getenv("CLIP_SECONDS", "3"))
BLOCKSIZE = int(os.getenv("BLOCKSIZE", "4096"))
LATENCY = os.getenv("LATENCY", "high")
MIN_GAP = float(os.getenv("MIN_GAP_SECONDS", "1.5"))
MIC_SPEC = os.getenv("MIC_DEVICE", "")

# --- Helper Functions (Unchanged) ---
def resolve_device(spec):
    """Find a sounddevice input device index from a string specifier."""
    if spec.strip().isdigit():
        return int(spec)
    s = spec.lower()
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and s in d["name"].lower():
            return i
    raise ValueError(f"No input device matching '{spec}'")

def rms_dbfs(x):
    """Calculate the RMS level of an int16 audio chunk in dBFS."""
    x = x.astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(x**2) + 1e-12)
    return 20 * np.log10(rms + 1e-12)

def write_wav(path, sr, x):
    """Write a 16-bit mono PCM WAV file."""
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x.tobytes())

# --- THE CONSUMER THREAD ---
def processing_thread_func(q: queue.Queue):
    """
    This function runs in a separate thread.
    It waits for audio chunks from the queue and processes them.
    """
    print("[recorder] Processing thread started.", flush=True)
    clip = []
    last_write = 0
    triggered = False
    dbfs_gate = -(100 - THRESH_DB) # Map threshold (e.g., 65dB) to a dBFS level

    while True:
        try:
            # Block and wait for a chunk from the producer
            mono_chunk = q.get()

            db = rms_dbfs(mono_chunk)

            if db >= dbfs_gate:  # If loud enough
                clip.append(mono_chunk)
                if not triggered:
                    triggered = True
            elif triggered and (time.time() - last_write) >= MIN_GAP:
                # --- Finalize and write the clip after a quiet period ---
                if not clip:
                    # Guard against empty clips if logic gets complex
                    triggered = False
                    continue

                # Concatenate all chunks and convert to int16
                y = np.concatenate(clip)

                # Pad or truncate to the required clip length
                target_len = int(CLIP_SECONDS * SAMPLE_RATE)
                if len(y) < target_len:
                    pad = np.zeros(target_len - len(y), dtype=np.int16)
                    y = np.concatenate([y, pad])
                else:
                    y = y[:target_len]

                # Generate filename and path
                ts = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
                path = os.path.join(DATA_DIR, f"{ts}_{SITE_ID}.wav")

                # Write the WAV file
                write_wav(path, SAMPLE_RATE, y)
                last_write = time.time()
                print(f"[recorder] Wrote {path} ({len(y)/SAMPLE_RATE:.1f}s)", flush=True)

                # Reset state for the next event
                clip = []
                triggered = False

        except Exception as e:
            print(f"[recorder] Error in processing thread: {e}", flush=True)


def main():
    """Main function to set up and run the recorder."""
    device = resolve_device(MIC_SPEC) if MIC_SPEC else None
    print(f"[recorder] Cfg: device={MIC_SPEC or device}, rate={SAMPLE_RATE}, "
          f"threshold={THRESH_DB}dB, clip={CLIP_SECONDS}s, block={BLOCKSIZE}", flush=True)
    
    os.makedirs(DATA_DIR, exist_ok=True)

    # The thread-safe queue for passing data between threads
    data_queue = queue.Queue()

    # --- THE PRODUCER (Callback) ---
    def audio_callback(indata, frames, time_info, status):
        """
        This is the real-time audio callback.
        It must be as fast as possible to avoid overflows.
        Its only job is to put the audio data onto the queue.
        """
        if status.input_overflow:
            print("! Input overflow detected", flush=True)
        try:
            # Put a copy of the mono audio data onto the queue
            data_queue.put(indata[:, 0].copy())
        except Exception as e:
            print(f"[recorder] Error in audio callback: {e}", flush=True)

    # --- Start the Consumer Thread ---
    consumer_thread = threading.Thread(
        target=processing_thread_func,
        args=(data_queue,),
        daemon=True  # A daemon thread exits when the main program exits
    )
    consumer_thread.start()

    # --- Start the Audio Stream ---
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=BLOCKSIZE,
            latency=LATENCY,
            device=device,
            callback=audio_callback
        ):
            print("[recorder] Listening for noise events...")
            # The main thread just needs to stay alive
            while True:
                time.sleep(10)
    except Exception as e:
        print(f"[recorder] Fatal error: {e}", flush=True)
        # Exit if the stream fails to start
        exit(1)

if __name__ == "__main__":
    main()
