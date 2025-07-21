#!/bin/bash

# LPR Application Deployment Script
# This script sets up the LPR application on a Linux server

set -e  # Exit on any error

echo "🚀 Starting LPR Application Deployment..."

# Configuration
APP_NAME="lpr-app"
DEPLOY_DIR="/opt/lpr-app"
DOCKER_IMAGE="lpr-app:latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root or with sudo
check_permissions() {
    if [[ $EUID -eq 0 ]]; then
        print_status "Running with root permissions ✓"
    else
        print_error "Please run with sudo: sudo ./deploy.sh"
        exit 1
    fi
}

# Install Docker if not present
install_docker() {
    if ! command -v docker &> /dev/null; then
        print_status "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        systemctl start docker
        systemctl enable docker
        usermod -aG docker $SUDO_USER 2>/dev/null || true
    else
        print_status "Docker already installed ✓"
    fi
}

# Install Docker Compose if not present
install_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        print_status "Installing Docker Compose..."
        curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        chmod +x /usr/local/bin/docker-compose
    else
        print_status "Docker Compose already installed ✓"
    fi
}

# Create deployment directory
setup_directory() {
    print_status "Setting up deployment directory..."
    mkdir -p $DEPLOY_DIR
    cd $DEPLOY_DIR
    
    # Create required directories
    mkdir -p output_dir debug_plates logs
    
    # Set proper permissions
    chown -R $SUDO_USER:$SUDO_USER $DEPLOY_DIR 2>/dev/null || true
    chmod -R 755 $DEPLOY_DIR
}

# Load Docker image if tar file exists
load_docker_image() {
    if [ -f "lpr-app.tar" ]; then
        print_status "Loading Docker image from tar file..."
        docker load -i lpr-app.tar
        print_status "Docker image loaded successfully ✓"
    else
        print_warning "lpr-app.tar not found. Make sure to transfer the image file or pull from registry."
    fi
}

# Create systemd service for auto-start
create_systemd_service() {
    print_status "Creating systemd service..."
    
    cat > /etc/systemd/system/lpr-app.service << EOF
[Unit]
Description=LPR Application
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$DEPLOY_DIR
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable lpr-app.service
    print_status "Systemd service created and enabled ✓"
}

# Start the application
start_application() {
    if [ -f "docker-compose.yml" ]; then
        print_status "Starting LPR application..."
        docker-compose up -d
        print_status "Application started successfully ✓"
        
        # Show status
        echo ""
        print_status "Application Status:"
        docker-compose ps
        
        echo ""
        print_status "🎉 Deployment completed successfully!"
        print_status "Web interface will be available at: http://$(hostname -I | awk '{print $1}'):8080"
        print_status "To view logs: docker-compose logs -f"
        print_status "To stop: docker-compose down"
        print_status "To restart: systemctl restart lpr-app"
    else
        print_error "docker-compose.yml not found. Please copy it to $DEPLOY_DIR"
        exit 1
    fi
}

# Main deployment flow
main() {
    print_status "LPR Application Deployment Starting..."
    
    check_permissions
    install_docker
    install_docker_compose
    setup_directory
    load_docker_image
    create_systemd_service
    start_application
}

# Run main function
main "$@"
