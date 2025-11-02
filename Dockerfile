## Multi-stage Dockerfile to reduce final image size
## Builder stage: install build tools, build wheels / install packages into an isolated prefix
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tzdata \
    libgomp1 \
    libgcc-s1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /install

COPY requirements-minimal.txt ./

# Build wheel files (cache) and install into /install/python
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip wheel --no-cache-dir --wheel-dir /install/wheels -r requirements-minimal.txt && \
    pip install --no-cache-dir --no-index --find-links /install/wheels -r requirements-minimal.txt --target /install/python

## Runtime stage: slim image with only runtime system libs (no build tools)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Kolkata

# Install only runtime system libraries required by OpenCV / YOLO / ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgcc-s1 \
    libgl1 \
    libfontconfig1 \
    libice6 \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Configure timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install/python /usr/local/lib/python3.11/site-packages

# Copy application code
COPY . .

# Create directories and set ownership for non-root user
RUN mkdir -p output_dir debug_plates /app/logs && \
    chmod +x main.py || true && \
    useradd -m -u 1000 lpruser || true && \
    mkdir -p /app/vendor && \
    chown -R lpruser:lpruser /app || true

# Copy entrypoint script and make executable
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh && chown lpruser:lpruser /usr/local/bin/entrypoint.sh || true

USER lpruser

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

CMD ["python", "main.py"]
