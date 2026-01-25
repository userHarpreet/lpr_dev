# LPR Application

This is a License Plate Recognition (LPR) application containerized with Docker.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Quick Start (Docker Compose)

The easiest way to run the application is using Docker Compose. This handles building the image and setting up the necessary volume mounts for persistence.

1.  **Build and Start**:
    ```bash
    docker-compose up --build
    ```
    *Note: This configuration now includes `platform: linux/amd64` to ensure compatibility on ARM devices (e.g., Apple Silicon M1/M2/M3) via emulation.*

2.  **Run in Background**:
    ```bash
    docker-compose up -d --build
    ```

3.  **Stop the Application**:
    ```bash
    docker-compose down
    ```

## Manual Run (Docker CLI)

If you prefer to use the Docker CLI directly:

1.  **Build the Image**:
    ```bash
    docker build -t lpr-app .
    ```

2.  **Run the Container**:
    
    *Important: You must mount the `data` directory to persist the database.*

    **Windows (PowerShell):**
    ```powershell
    docker run -it --rm -v ${PWD}/data:/app/data lpr-app
    ```

    **Linux/macOS:**
    ```bash
    docker run -it --rm -v $(pwd)/data:/app/data lpr-app
    ```

## Persistence

The application stores its SQLite database in the `data` directory relative to the project root.
- **Docker Compose**: Automatically maps `./data` to `/app/data`.
- **Manual Run**: You must manually specify `-v ${PWD}/data:/app/data`.

## Troubleshooting

### ARM / Apple Silicon Users (exec format error)

If you see an `exec format error` during build, it means your Docker environment cannot emulate the required x86_64 architecture. To fix this, you need to install the QEMU emulators:

```bash
docker run --privileged --rm tonistiigi/binfmt --install all
```

After running this command once, try the build again.
