#!/usr/bin/env pwsh

# LPR Docker Build and Management Script
# This script helps build, run, and manage the LPR Docker container

param(
    [Parameter(Position=0)]
    [ValidateSet("build", "run", "stop", "logs", "shell", "clean", "help")]
    [string]$Action = "help",
    
    [Parameter()]
    [string]$Tag = "lpr-app:latest",
    
    [Parameter()]
    [switch]$Force
)

function Show-Help {
    Write-Host @"
LPR Docker Management Script

Usage: .\build.ps1 [action] [options]

Actions:
  build     Build the Docker image
  run       Run the container with docker-compose
  stop      Stop the container
  logs      Show container logs
  shell     Open shell in running container
  clean     Clean up Docker resources
  help      Show this help message

Options:
  -Tag      Docker image tag (default: lpr-app:latest)
  -Force    Force rebuild without cache

Examples:
  .\build.ps1 build              # Build the Docker image
  .\build.ps1 build -Force       # Build without cache
  .\build.ps1 run                # Start the application
  .\build.ps1 logs               # View logs
  .\build.ps1 shell              # Access container shell
  .\build.ps1 stop               # Stop containers
  .\build.ps1 clean              # Clean up resources

"@
}

function Build-Image {
    Write-Host "Building Docker image: $Tag" -ForegroundColor Green
    
    $buildArgs = @("build", "-t", $Tag, ".")
    
    if ($Force) {
        $buildArgs += "--no-cache"
        Write-Host "Building without cache..." -ForegroundColor Yellow
    }
    
    & docker @buildArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Build completed successfully!" -ForegroundColor Green
        & docker images $Tag
    } else {
        Write-Host "Build failed!" -ForegroundColor Red
        exit 1
    }
}

function Run-Container {
    Write-Host "Starting LPR application with docker-compose..." -ForegroundColor Green
    
    # Create necessary directories
    if (!(Test-Path "output_dir")) { New-Item -ItemType Directory -Path "output_dir" }
    if (!(Test-Path "debug_plates")) { New-Item -ItemType Directory -Path "debug_plates" }
    
    & docker-compose up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Application started successfully!" -ForegroundColor Green
        Write-Host "Web interface available at: http://localhost:8080" -ForegroundColor Cyan
        Write-Host "Use '.\build.ps1 logs' to view application logs" -ForegroundColor Yellow
    } else {
        Write-Host "Failed to start application!" -ForegroundColor Red
    }
}

function Stop-Container {
    Write-Host "Stopping LPR application..." -ForegroundColor Yellow
    & docker-compose down
    Write-Host "Application stopped." -ForegroundColor Green
}

function Show-Logs {
    Write-Host "Showing container logs..." -ForegroundColor Green
    & docker-compose logs -f lpr
}

function Open-Shell {
    Write-Host "Opening shell in LPR container..." -ForegroundColor Green
    & docker-compose exec lpr /bin/bash
}

function Clean-Resources {
    Write-Host "Cleaning up Docker resources..." -ForegroundColor Yellow
    
    if ($Force) {
        Write-Host "Performing force cleanup..." -ForegroundColor Red
        & docker-compose down -v --remove-orphans
        & docker system prune -f
        & docker volume prune -f
    } else {
        & docker-compose down --remove-orphans
        & docker image prune -f
    }
    
    Write-Host "Cleanup completed." -ForegroundColor Green
}

# Main script logic
switch ($Action) {
    "build" { Build-Image }
    "run" { Run-Container }
    "stop" { Stop-Container }
    "logs" { Show-Logs }
    "shell" { Open-Shell }
    "clean" { Clean-Resources }
    "help" { Show-Help }
    default { Show-Help }
}
