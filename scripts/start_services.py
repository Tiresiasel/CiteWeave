#!/usr/bin/env python3
"""
Start CiteWeave services (Qdrant + GROBID) using docker-compose.
"""

import subprocess
import sys
import time
import requests
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_docker():
    """Check if Docker is available."""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logging.info(f"Docker available: {result.stdout.strip()}")
            return True
        else:
            logging.error("Docker is not available")
            return False
    except FileNotFoundError:
        logging.error("Docker not found. Please install Docker first.")
        return False

def check_docker_compose():
    """Check if docker-compose is available."""
    try:
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logging.info(f"Docker Compose available: {result.stdout.strip()}")
            return True
        else:
            logging.error("Docker Compose is not available")
            return False
    except FileNotFoundError:
        logging.error("Docker Compose not found. Please install Docker Compose first.")
        return False

def start_services():
    """Start both Qdrant and GROBID services."""
    logging.info("Starting CiteWeave services (Qdrant + GROBID)...")
    
    try:
        # Start services in detached mode
        result = subprocess.run(
            ['docker-compose', 'up', '-d'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent  # Run from project root
        )
        
        if result.returncode == 0:
            logging.info("Services started successfully!")
            logging.info(result.stdout)
            return True
        else:
            logging.error(f"Failed to start services: {result.stderr}")
            return False
            
    except Exception as e:
        logging.error(f"Error starting services: {e}")
        return False

def wait_for_service(url: str, service_name: str, max_retries: int = 30):
    """Wait for a service to be ready."""
    logging.info(f"Waiting for {service_name} to be ready...")
    
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logging.info(f"✅ {service_name} is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if i < max_retries - 1:
            logging.info(f"⏳ {service_name} not ready yet, retrying in 2 seconds... ({i+1}/{max_retries})")
            time.sleep(2)
    
    logging.error(f"❌ {service_name} failed to start within {max_retries * 2} seconds")
    return False

def check_services():
    """Check if both services are running."""
    logging.info("Checking service status...")
    
    # Check Qdrant
    qdrant_ready = wait_for_service("http://localhost:6333/health", "Qdrant", max_retries=15)
    
    # Check GROBID
    grobid_ready = wait_for_service("http://localhost:8070/api/isalive", "GROBID", max_retries=30)
    
    if qdrant_ready and grobid_ready:
        logging.info("🎉 All services are ready!")
        return True
    else:
        logging.error("❌ Some services failed to start")
        return False

def show_status():
    """Show the status of running services."""
    logging.info("Checking service status...")
    
    try:
        result = subprocess.run(
            ['docker-compose', 'ps'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            logging.info("Service Status:")
            print(result.stdout)
        else:
            logging.error(f"Failed to get service status: {result.stderr}")
            
    except Exception as e:
        logging.error(f"Error checking service status: {e}")

def main():
    """Main function."""
    print("🚀 CiteWeave Services Manager")
    print("=" * 40)
    
    # Check prerequisites
    if not check_docker():
        sys.exit(1)
    
    if not check_docker_compose():
        sys.exit(1)
    
    # Start services
    if not start_services():
        sys.exit(1)
    
    # Wait for services to be ready
    if not check_services():
        logging.error("Services failed to start properly")
        show_status()
        sys.exit(1)
    
    # Show final status
    show_status()
    
    print("\n🎯 Services are ready!")
    print("📊 Qdrant: http://localhost:6333")
    print("📚 GROBID: http://localhost:8070")
    print("\nTo stop services, run: docker-compose down")

if __name__ == "__main__":
    main() 