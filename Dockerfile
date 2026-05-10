# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# playercount — multi-stage image
#
# Two base targets:
#   - cpu : python:3.11-slim                              (default)
#   - gpu : nvidia/cuda:12.4.0-runtime-ubuntu22.04        (--build-arg BASE=gpu)
#
# Build:
#     docker build -t playercount:cpu .
#     docker build -t playercount:gpu --build-arg BASE=gpu .
#
# Run (cpu):
#     docker run --rm -p 8000:8000 \
#         -v $PWD/models:/app/models:ro \
#         -v $PWD/data:/app/data:ro \
#         playercount:cpu
# Run (gpu):
#     docker run --rm --gpus all -p 8000:8000 ... playercount:gpu
# ---------------------------------------------------------------------------

ARG BASE=cpu

# ---------- common builder ---------------------------------------------------

FROM python:3.11-slim AS builder-cpu
ARG BASE
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip \
 && pip wheel --wheel-dir /wheels ".[cpu]"

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS builder-gpu
ARG BASE
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        build-essential \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip \
 && pip wheel --wheel-dir /wheels ".[gpu]"

# Pick the right builder for the BASE arg.
FROM builder-${BASE} AS builder

# ---------- runtime ---------------------------------------------------------

FROM python:3.11-slim AS runtime-cpu
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --shell /usr/sbin/nologin app

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS runtime-gpu
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3.11 python3-pip \
        ffmpeg libgl1 libglib2.0-0 \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --shell /usr/sbin/nologin app

FROM runtime-${BASE} AS runtime
ARG BASE

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PLAYERCOUNT_LOG_JSON=true \
    PLAYERCOUNT_WEIGHTS_DIR=/app/models

WORKDIR /app

# Install pre-built wheels from the builder stage (no network at runtime).
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels /wheels/*.whl \
 && rm -rf /wheels

# App code
COPY --chown=app:app src ./src
COPY --chown=app:app configs ./configs
COPY --chown=app:app pyproject.toml README.md ./

# Make the package importable without re-running pip (wheel already installed).
ENV PYTHONPATH=/app/src

USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8000/healthz', timeout=2).is_success else 1)"

CMD ["python", "-m", "uvicorn", "playercount.api.main:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000"]
