#!/usr/bin/env python3
"""
Setup script for CiteWeave data directories
Creates the proper directory structure for both local development and Docker deployment
"""

import os
import sys
from pathlib import Path

def setup_data_directories():
    """Create the proper data directory structure"""
    
    # Get project root
    project_root = Path(__file__).parent.parent
    print(f"Setting up data directories in: {project_root}")
    
    # Define directories to create
    directories = [
        "data",
        "data/papers",
        "data/vector_index",
        "data/logs",
        "logs",
        "config"
    ]
    
    # Create directories
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")
    
    # Create .gitkeep files to preserve empty directories
    gitkeep_dirs = [
        "data/papers",
        "data/vector_index", 
        "data/logs",
        "logs"
    ]
    
    for directory in gitkeep_dirs:
        gitkeep_file = project_root / directory / ".gitkeep"
        if not gitkeep_file.exists():
            gitkeep_file.touch()
            print(f"✅ Created: {gitkeep_file}")
    
    # Create sample .env file if it doesn't exist
    env_file = project_root / ".env"
    if not env_file.exists():
        env_content = """# CiteWeave Environment Configuration
# Data directory for persistent storage
CITEWEAVE_DATA_DIR=data

# API configuration
CITEWEAVE_API_HOST=0.0.0.0
CITEWEAVE_API_PORT=31415

# Database URLs (for local development)
QDRANT_URL=http://localhost:6333
GROBID_URL=http://localhost:8070
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=12345678

# Database file extensions (using .sqlite for clarity)
# All SQLite databases use .sqlite extension for consistency

# For Docker deployment, these will be overridden by docker-compose.yml
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✅ Created: {env_file}")
    
    print("\n🎉 Data directory setup completed!")
    print("\n📁 Directory structure:")
    print("  data/")
    print("  ├── papers/          # PDF papers and processed data")
    print("  ├── vector_index/    # Qdrant vector data")
    print("  └── logs/            # Application logs")
    print("  logs/                # Query trace logs")
    print("  config/              # Configuration files")
    
    print("\n💡 Next steps:")
    print("  1. For local development: Use the created directories")
    print("  2. For Docker: Data will be mounted from these directories")
    print("  3. Check .env file and modify as needed")

if __name__ == "__main__":
    setup_data_directories()
