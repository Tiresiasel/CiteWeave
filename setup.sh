#!/bin/bash
# CiteWeave Main Setup Control Script
# This script orchestrates all setup operations for CiteWeave

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[SETUP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
🚀 CiteWeave Setup Control Script

Usage: ./setup.sh [OPTION]

Options:
  (no option)        Full setup: directories + deployment config
  --directories      Setup data directories only
  --deployment       Setup deployment configuration only
  --services         Setup database services (Neo4j + Qdrant)
  --full             Full setup with all options
  --help             Show this help message

Examples:
  ./setup.sh                    # Full setup (default)
  ./setup.sh --directories      # Create data directories only
  ./setup.sh --deployment       # Configure deployment only
  ./setup.sh --services         # Setup database services only

EOF
}

# Function to setup data directories
setup_directories() {
    print_status "Setting up data directories..."
    
    if [[ -f "scripts/setup_data_directories.py" ]]; then
        python3 scripts/setup_data_directories.py
        print_success "Data directories setup completed"
    else
        print_error "setup_data_directories.py not found"
        return 1
    fi
}

# Function to setup deployment configuration
setup_deployment() {
    print_status "Setting up deployment configuration..."
    
    if [[ -f "scripts/setup_deployment.sh" ]]; then
        # Check if .env exists and ask for confirmation
        if [[ -f ".env" ]]; then
            print_warning ".env file already exists. Do you want to overwrite it? (y/N)"
            read -r response
            if [[ ! "$response" =~ ^[Yy]$ ]]; then
                print_status "Skipping deployment configuration"
                return 0
            fi
        fi
        
        ./scripts/setup_deployment.sh
        print_success "Deployment configuration completed"
    else
        print_error "setup_deployment.sh not found"
        return 1
    fi
}

# Function to setup database services
setup_services() {
    print_status "Setting up database services..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        return 1
    fi
    
    # Check if services are already running
    if docker ps --format "table {{.Names}}" | grep -q "citeweave-"; then
        print_warning "CiteWeave services are already running. Stopping them first..."
        ./run.sh --no-update-watch-map --no-rebuild 2>/dev/null || true
    fi
    
    print_status "Starting database services..."
    ./run.sh --rebuild
    
    # Wait for services to be healthy
    print_status "Waiting for services to be healthy..."
    sleep 10
    
    # Check service health
    if curl -s http://localhost:31415/api/v1/health >/dev/null 2>&1; then
        print_success "All services are running and healthy"
    else
        print_warning "Services may still be starting up. Check with: ./run.sh"
    fi
}

# Function to perform full setup
full_setup() {
    print_status "Performing full setup..."
    
    setup_directories
    setup_deployment
    setup_services
    
    print_success "Full setup completed successfully!"
    
    echo ""
    echo "🎉 CiteWeave is now ready!"
    echo ""
    echo "📋 Next steps:"
    echo "  1. Add your documents to the documents/ directory"
    echo "  2. Access CiteWeave at: http://localhost:31415"
    echo "  3. Check logs with: docker logs citeweave-app"
    echo ""
    echo "📁 Directory structure:"
    echo "  data/           # Persistent data storage"
    echo "  logs/           # Query trace logs"
    echo "  documents/      # Documents to process"
    echo "  .env            # Environment configuration"
}

# Main execution logic
main() {
    echo "🚀 CiteWeave Setup Control Script"
    echo "================================"
    echo ""
    
    case "${1:-}" in
        --directories)
            setup_directories
            ;;
        --deployment)
            setup_deployment
            ;;
        --services)
            setup_services
            ;;
        --full)
            full_setup
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        "")
            # No arguments - perform full setup
            full_setup
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
