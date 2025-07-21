# LPR Application Deployment Guide

This guide will help you deploy your LPR application to a Linux server.

## Quick Start

### Prerequisites
- Linux server (Ubuntu 18.04+ or CentOS 7+ recommended)
- SSH access to the server
- At least 8GB RAM and 20GB free disk space

### Files to Transfer to Server
1. `lpr-app.tar` (Docker image - 1.2GB)
2. `docker-compose.yml`
3. `requirements/` folder (contains models and config)
4. `deploy.sh` (deployment script)

## Deployment Methods

### Method 1: Automated Deployment (Recommended)

1. **Transfer files to your server:**
   ```bash
   # Using SCP (replace with your server details)
   scp lpr-app.tar docker-compose.yml deploy.sh user@your-server:/home/user/
   scp -r requirements user@your-server:/home/user/
   ```

2. **Connect to your server and run deployment:**
   ```bash
   ssh user@your-server
   chmod +x deploy.sh
   sudo ./deploy.sh
   ```

3. **Access your application:**
   - Web interface: `http://your-server-ip:8080`
   - The application will auto-start on system boot

### Method 2: Manual Deployment

1. **Install Docker and Docker Compose:**
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo systemctl start docker
   sudo systemctl enable docker
   sudo usermod -aG docker $USER

   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

2. **Set up application directory:**
   ```bash
   sudo mkdir -p /opt/lpr-app
   cd /opt/lpr-app
   
   # Copy your files here
   sudo cp /path/to/your/files/* ./
   sudo cp -r /path/to/requirements ./
   
   # Create required directories
   sudo mkdir -p output_dir debug_plates logs
   sudo chown -R $USER:$USER /opt/lpr-app
   ```

3. **Load Docker image:**
   ```bash
   docker load -i lpr-app.tar
   ```

4. **Start the application:**
   ```bash
   docker-compose up -d
   ```

### Method 3: Docker Hub Deployment (Advanced)

1. **Push image to Docker Hub (from your development machine):**
   ```bash
   docker tag lpr-app:latest yourusername/lpr-app:latest
   docker login
   docker push yourusername/lpr-app:latest
   ```

2. **Update docker-compose.yml on server:**
   ```yaml
   services:
     lpr:
       image: yourusername/lpr-app:latest  # Use your Docker Hub image
       # ... rest of configuration
   ```

3. **Deploy on server:**
   ```bash
   docker-compose pull
   docker-compose up -d
   ```

## Server Requirements

### Minimum Requirements:
- **OS**: Ubuntu 18.04+, CentOS 7+, or similar Linux distribution
- **CPU**: 2 cores
- **RAM**: 4GB (8GB recommended)
- **Storage**: 20GB free space
- **Network**: Port 8080 accessible (configure firewall if needed)

### Optimal Requirements:
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: SSD with 50GB+ free space
- **GPU**: NVIDIA GPU with Docker GPU support (optional, for better performance)

## Firewall Configuration

If your server has a firewall, allow port 8080:

```bash
# Ubuntu (ufw)
sudo ufw allow 8080

# CentOS/RHEL (firewalld)
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload

# iptables
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
```

## Managing the Application

### View logs:
```bash
cd /opt/lpr-app
docker-compose logs -f lpr
```

### Stop the application:
```bash
docker-compose down
```

### Start the application:
```bash
docker-compose up -d
```

### Restart the application:
```bash
docker-compose restart
```

### Update the application:
```bash
# If using Docker Hub
docker-compose pull
docker-compose up -d

# If using tar file
docker load -i new-lpr-app.tar
docker-compose up -d
```

## Monitoring

### Check container status:
```bash
docker-compose ps
```

### View resource usage:
```bash
docker stats lpr-container
```

### Check system resources:
```bash
htop  # or top
df -h  # disk usage
```

## Troubleshooting

### Application won't start:
```bash
# Check logs
docker-compose logs lpr

# Check if ports are in use
netstat -tlnp | grep 8080

# Restart Docker service
sudo systemctl restart docker
```

### Out of memory errors:
```bash
# Check memory usage
free -h

# Increase swap space or upgrade RAM
# Adjust memory limits in docker-compose.yml
```

### Permission errors:
```bash
# Fix file permissions
sudo chown -R $USER:$USER /opt/lpr-app
sudo chmod -R 755 /opt/lpr-app
```

## Security Considerations

1. **Change default ports** if exposed to internet
2. **Set up reverse proxy** (nginx) for production
3. **Enable HTTPS** with SSL certificates
4. **Restrict network access** to trusted IPs
5. **Regular backups** of configuration and models
6. **Keep Docker updated** for security patches

## Backup and Recovery

### Backup:
```bash
# Backup configuration and data
tar -czf lpr-backup-$(date +%Y%m%d).tar.gz /opt/lpr-app/

# Backup Docker image
docker save lpr-app:latest | gzip > lpr-image-backup.tar.gz
```

### Recovery:
```bash
# Restore files
tar -xzf lpr-backup-YYYYMMDD.tar.gz -C /

# Load Docker image
gunzip -c lpr-image-backup.tar.gz | docker load
```

## Support

If you encounter issues:
1. Check the logs first: `docker-compose logs lpr`
2. Verify all files are transferred correctly
3. Ensure server meets minimum requirements
4. Check network connectivity and firewall settings

For additional help, provide:
- Error logs
- Server specifications
- Docker version: `docker --version`
- Docker Compose version: `docker-compose --version`
