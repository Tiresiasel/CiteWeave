#!/usr/bin/env python3
"""
Setup script for Qdrant server with data migration
"""

import os
import sys
import json
import shutil
import subprocess
import time
from pathlib import Path

def check_docker():
    """Check if Docker is available"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker is available")
            return True
        else:
            print("❌ Docker is not available")
            return False
    except FileNotFoundError:
        print("❌ Docker is not installed")
        return False

def check_docker_compose():
    """Check if Docker Compose is available"""
    try:
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker Compose is available")
            return True
        else:
            print("❌ Docker Compose is not available")
            return False
    except FileNotFoundError:
        print("❌ Docker Compose is not installed")
        return False

def start_qdrant_server():
    """Start Qdrant server using Docker Compose"""
    print("🚀 Starting Qdrant server...")
    
    try:
        # Start the server
        result = subprocess.run(['docker-compose', 'up', '-d', 'qdrant'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Qdrant server started successfully")
            return True
        else:
            print(f"❌ Failed to start Qdrant server: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error starting Qdrant server: {e}")
        return False

def wait_for_qdrant_ready():
    """Wait for Qdrant server to be ready"""
    print("⏳ Waiting for Qdrant server to be ready...")
    
    import requests
    
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:6333/", timeout=5)
            if response.status_code == 200:
                print("✅ Qdrant server is ready!")
                return True
        except:
            pass
        
        print(f"   Attempt {attempt + 1}/{max_attempts}...")
        time.sleep(2)
    
    print("❌ Qdrant server did not become ready in time")
    return False

def migrate_existing_data():
    """Migrate existing vector index data to Qdrant server"""
    local_index_path = Path("./data/vector_index")
    
    if not local_index_path.exists():
        print("ℹ️  No existing vector index found, skipping migration")
        return True
    
    print("📦 Migrating existing vector index data...")
    
    try:
        # Create backup
        backup_path = Path("./data/vector_index_backup")
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.copytree(local_index_path, backup_path)
        print(f"✅ Created backup at {backup_path}")
        
        # Copy data to Docker volume
        docker_volume_path = Path("./qdrant_storage")
        if docker_volume_path.exists():
            shutil.rmtree(docker_volume_path)
        
        # Create the directory structure
        docker_volume_path.mkdir(parents=True, exist_ok=True)
        
        # Copy the storage directory
        local_storage = local_index_path / "storage"
        if local_storage.exists():
            docker_storage = docker_volume_path / "storage"
            shutil.copytree(local_storage, docker_storage)
            print("✅ Migrated vector index data to Docker volume")
        else:
            print("ℹ️  No storage data found in local index")
        
        return True
        
    except Exception as e:
        print(f"❌ Error migrating data: {e}")
        return False

def create_collections():
    """Create collections in Qdrant server"""
    print("🗂️  Creating collections in Qdrant server...")
    
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        client = QdrantClient(host="localhost", port=6333)
        
        collections = ["sentences", "paragraphs", "sections", "citations"]
        
        for collection_name in collections:
            try:
                client.get_collection(collection_name)
                print(f"✅ Collection '{collection_name}' already exists")
            except:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                print(f"✅ Created collection '{collection_name}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating collections: {e}")
        return False

def main():
    """Main setup function"""
    print("🔧 Setting up Qdrant server for CiteWeave...")
    
    # Check prerequisites
    if not check_docker():
        print("Please install Docker first: https://docs.docker.com/get-docker/")
        return False
    
    if not check_docker_compose():
        print("Please install Docker Compose first: https://docs.docker.com/compose/install/")
        return False
    
    # Migrate existing data
    if not migrate_existing_data():
        print("Failed to migrate existing data")
        return False
    
    # Start Qdrant server
    if not start_qdrant_server():
        print("Failed to start Qdrant server")
        return False
    
    # Wait for server to be ready
    if not wait_for_qdrant_ready():
        print("Failed to start Qdrant server")
        return False
    
    # Create collections
    if not create_collections():
        print("Failed to create collections")
        return False
    
    print("\n🎉 Qdrant server setup complete!")
    print("📊 Server is running at: http://localhost:6333")
    print("🔗 API endpoint: http://localhost:6333/api/v1")
    print("📚 Web UI: http://localhost:6333/dashboard")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 