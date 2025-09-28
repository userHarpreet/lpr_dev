FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
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
    git \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set timezone environment variable and configure system timezone to
# ensure processes inside the container use the specified TZ.
ENV TZ=Asia/Kolkata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

COPY requirements-minimal.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-minimal.txt

COPY . .

RUN mkdir -p output_dir debug_plates

RUN chmod +x main.py && \
    useradd -m -u 1000 lpruser && \
    chown -R lpruser:lpruser /app

USER lpruser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

CMD ["python", "main.py"]
