FROM python:3.11-slim

# System deps for OpenCV. The headless wheel still dlopen's libGL/libglib at
# import time, so both are required even though there is no GUI.
# rasterio (app/pipeline/loader.py) needs no apt packages here: its manylinux
# wheel bundles its own GDAL build, including the PDS4 driver. Only add
# gdal-bin/libgdal-dev if a future pip install of it has to build from source
# (e.g. no wheel for the base image's Python version — see the pin note in
# requirements.txt) — confirm by importing rasterio in the built image first.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY app/ ./app/

RUN mkdir -p /data
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
