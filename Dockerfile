# Use Ubuntu 22.04 as base image for better compatibility with PaddlePaddle and OpenCV
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libc6-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    libxext6 \
    libgl1-mesa-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy minimal requirements first to leverage Docker cache and avoid conflicts
COPY requirements-minimal.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements-minimal.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p output_dir debug_plates

# Set permissions
RUN chmod +x main.py

# Create a non-root user for security
RUN useradd -m -u 1000 lpruser && chown -R lpruser:lpruser /app
USER lpruser

# Expose port if needed (for web interface)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python3", "main.py"]
