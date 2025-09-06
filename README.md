# UrbanSound Pi — Suburban Noise Monitor

**Goal:** A small, repeatable, edge stack that records and classifies urban sounds on a **Raspberry Pi 4** and exposes a simple dashboard. Optimised for **NSW Noise Policy for Industry (NPFI)** monitoring practice.

- **Recorder** (Python): captures WAV clips when sound exceeds a threshold
- **Classifier** (Python/TFLite): tags clips; optional **Coral TPU**
- **Dashboard** (Dash): lists clips & labels, basic plots
- **Kubernetes**: single-node **MicroK8s** on the Pi; local **registry**; hostPath PVC to **USB stick** for data

> ⚠️ **Compliance note:** This is a helpful monitor, **not** a calibrated SLM. If you need NPfI-compliant assessments, use a **Class 1/2 instrument** and EPA methods. This repo can compute helpful metrics (LAeq/LA90) but must be **calibrated** to report dB(A) SPL.

---

## NSW suburban targets (for your dashboard overlays)

From NSW NPFI for **residential – suburban (outdoor)**:

- **Day:** 55 dB LAeq,period  
- **Evening:** 45 dB LAeq,period  
- **Night:** 40 dB LAeq,period

Intrusiveness check: **LAeq,15min ≤ RBL + 5 dB** (RBL from LA90).  
Use these as **dashboard overlays**; the recorder’s trigger remains a practical dBFS threshold.

---

## Architecture

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/9976d9aa-27fb-40e7-8ba0-e8c4b7799791" />
