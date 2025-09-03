FROM python:3.10-slim-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy==1.26.4 paho-mqtt==2.1.0 \
 && (pip install --no-cache-dir tflite-runtime==2.13.0 || true)

WORKDIR /app
COPY app.py                /app/app.py
COPY assets/model.tflite   /app/model.tflite
COPY assets/labels.txt     /app/labels.txt

# fail fast if assets didn’t copy or are empty
RUN test -s /app/model.tflite && test -s /app/labels.txt

ENV PYTHONUNBUFFERED=1
CMD ["python","/app/app.py"]

