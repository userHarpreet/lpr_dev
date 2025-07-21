# LPR Docker Setup Guide

This guide explains how to containerize and run the License Plate Recognition (LPR) application using Docker.

## Prerequisites

- Docker Desktop installed and running
- Docker Compose installed (included with Docker Desktop)
- PowerShell (for Windows users) or Bash (for Linux/macOS)
- At least 4GB of available RAM
- At least 10GB of free disk space

## Quick Start

1. **Build the Docker image:**
   ```powershell
   .\build.ps1 build
   ```

2. **Run the application:**
   ```powershell
   .\build.ps1 run
   ```

3. **View logs:**
   ```powershell
   .\build.ps1 logs
   ```

4. **Stop the application:**
   ```powershell
   .\build.ps1 stop
   ```

## Docker Files Overview

### Core Docker Files

- **`Dockerfile`** - Main Docker image definition
- **`docker-compose.yml`** - Multi-container orchestration
- **`.dockerignore`** - Files to exclude from Docker build context
- **`nginx.conf`** - Web server configuration for serving HTML reports

### Management Scripts

- **`build.ps1`** - PowerShell script for Docker operations (Windows)
- **`build.sh`** - Bash script for Docker operations (Linux/macOS)

## Detailed Setup Instructions

### 1. Build the Docker Image

```powershell
# Standard build
.\build.ps1 build

# Force rebuild without cache (if you made changes)
.\build.ps1 build -Force

# Manual Docker build (alternative)
docker build -t lpr-app:latest .
```

### 2. Run the Application

The application uses Docker Compose for orchestration:

```powershell
# Start with compose (recommended)
.\build.ps1 run

# Manual compose commands
docker-compose up -d          # Start in background
docker-compose up             # Start with logs visible
docker-compose down           # Stop and remove containers
```

### 3. Access the Application

- **Web Interface:** http://localhost:8080
- **LPR Reports:** http://localhost:8080/output_dir/
- **Container Logs:** `.\build.ps1 logs` or `docker-compose logs -f lpr`

### 4. Monitor the Application

```powershell
# View real-time logs
.\build.ps1 logs

# Access container shell
.\build.ps1 shell

# Check container status
docker-compose ps

# View resource usage
docker stats lpr-container
```

## Configuration

### Environment Variables

You can override configuration settings using environment variables in `docker-compose.yml`:

```yaml
environment:
  - SHOW_LIVE=False
  - PLATE_CONF_MIN=0.65
  - VEHICLE_CONF_MIN=0.5
```

### Volume Mounts

The following directories are mounted as volumes:

- **`./output_dir`** - LPR detection results
- **`./debug_plates`** - Debug images
- **`./requirements`** - Configuration and model files (read-only)

### Resource Limits

Default resource limits (can be adjusted in `docker-compose.yml`):

- **CPU:** 2 cores max, 1 core reserved
- **Memory:** 4GB max, 2GB reserved

## Troubleshooting

### Common Issues

1. **Build Failures:**
   ```powershell
   # Clear Docker cache and rebuild
   .\build.ps1 build -Force
   
   # Check Docker daemon is running
   docker version
   ```

2. **Container Won't Start:**
   ```powershell
   # Check logs for errors
   .\build.ps1 logs
   
   # Verify file permissions
   docker-compose exec lpr ls -la /app
   ```

3. **High Memory Usage:**
   ```powershell
   # Monitor resource usage
   docker stats lpr-container
   
   # Adjust memory limits in docker-compose.yml
   ```

4. **Model Loading Issues:**
   ```powershell
   # Verify model files exist
   docker-compose exec lpr ls -la /app/requirements/
   
   # Check model file permissions
   docker-compose exec lpr python3 -c "import torch; print(torch.__version__)"
   ```

### Debugging Commands

```powershell
# Access container shell
docker-compose exec lpr /bin/bash

# View container configuration
docker inspect lpr-container

# Check network connectivity
docker-compose exec lpr ping google.com

# Test Python environment
docker-compose exec lpr python3 -c "import paddleocr; print('OK')"
```

## Performance Optimization

### For Better Performance:

1. **Allocate More Resources:**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '4.0'      # Increase CPU limit
         memory: 8G       # Increase memory limit
   ```

2. **Use GPU Support (if available):**
   ```yaml
   runtime: nvidia
   environment:
     - NVIDIA_VISIBLE_DEVICES=all
   ```

3. **Optimize Docker Build:**
   ```dockerfile
   # Use multi-stage builds
   FROM python:3.9-slim as builder
   # ... install dependencies
   
   FROM python:3.9-slim
   COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
   ```

## Maintenance

### Regular Tasks

```powershell
# Update Docker images
docker-compose pull

# Clean up unused resources
.\build.ps1 clean

# Force cleanup (removes all unused Docker resources)
.\build.ps1 clean -Force

# Backup important data
docker run --rm -v lpr_lpr-logs:/data -v ${PWD}:/backup ubuntu tar czf /backup/logs-backup.tar.gz -C /data .
```

### Log Management

Logs are automatically rotated with the following settings:
- Maximum file size: 10MB
- Maximum files: 3
- Total log storage: ~30MB

## Security Considerations

- Container runs as non-root user (`lpruser`)
- Read-only mounts for configuration files
- Network isolation through Docker networks
- Security headers configured in nginx
- No sensitive data in environment variables

## Support

For issues specific to:
- **Docker setup:** Check this README and Docker documentation
- **LPR application:** Check main README.md
- **Performance issues:** Monitor with `docker stats` and adjust resources

## Advanced Usage

### Custom Network Configuration

```yaml
networks:
  lpr-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Production Deployment

For production use:

1. Use external volumes for persistent storage
2. Configure proper log aggregation
3. Set up health checks and monitoring
4. Use secrets management for sensitive configuration
5. Configure backup strategies for model files and results

## Manual Commands Reference

```bash
# Build image manually
docker build -t lpr-app:latest .

# Run single container
docker run -d --name lpr-container -v ./output_dir:/app/output_dir lpr-app:latest

# View logs
docker logs -f lpr-container

# Execute commands in container
docker exec -it lpr-container python3 main.py

# Clean up
docker stop lpr-container
docker rm lpr-container
docker rmi lpr-app:latest
```
