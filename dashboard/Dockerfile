FROM python:3.10-slim-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir dash==2.17.1 pandas==2.2.2

WORKDIR /app
COPY app.py /app/app.py

ENV PYTHONUNBUFFERED=1
EXPOSE 5000
CMD ["python", "/app/app.py"]

