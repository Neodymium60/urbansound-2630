FROM python:3.10-slim-bullseye

# Audio libs for ALSA/PortAudio
RUN apt-get update && apt-get install -y --no-install-recommends \
      alsa-utils libasound2 libasound2-dev portaudio19-dev procps util-linux \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir numpy==1.26.4 sounddevice==0.4.6

WORKDIR /app
COPY app.py /app/app.py

ENV PYTHONUNBUFFERED=1
CMD ["python", "/app/app.py"]

