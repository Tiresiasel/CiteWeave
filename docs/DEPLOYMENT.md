# CiteWeave Deployment Guide

## 🚀 Quick Deployment

### Prerequisites
- Docker and Docker Compose installed
- Git (to clone the repository)

### Step 1: Clone and Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd CiteWeave

# Run the automated setup script
./scripts/setup_deployment.sh
```

### Step 2: Customize Configuration (Optional)
Edit the generated `.env` file to match your environment:
```bash
# Example for different document directory
DOCS_WATCH_DIRS=./my_documents,./research_papers

# Example for different ports
CITEWEAVE_API_PORT=8080
```

### Step 3: Start Services
```bash
# Start all services with rebuild
./run.sh --rebuild
```

### Step 4: Access CiteWeave
Open your browser and navigate to: `http://localhost:31415`

## 🔧 Manual Configuration

### Environment Variables
All configuration is done through environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CITEWEAVE_DATA_DIR` | `./data` | Host data directory |
| `CITEWEAVE_LOGS_DIR` | `./logs` | Host logs directory |
| `DOCS_WATCH_DIRS` | `./documents` | Document directories to watch |
| `CITEWEAVE_API_PORT` | `31415` | API server port |

### Directory Structure
```
CiteWeave/
├── data/           # Persistent data storage
├── logs/           # Query trace logs
├── documents/      # Documents to process
├── .env            # Environment configuration
└── docker-compose.yml
```

## 🌍 Cross-Platform Deployment

### Linux/macOS
```bash
# Use relative paths (recommended)
CITEWEAVE_DATA_DIR=./data
CITEWEAVE_LOGS_DIR=./logs
DOCS_WATCH_DIRS=./documents
```

### Windows
```bash
# Use Windows-style paths if needed
CITEWEAVE_DATA_DIR=./data
CITEWEAVE_LOGS_DIR=./logs
DOCS_WATCH_DIRS=./documents
```

### Custom Paths
```bash
# Use environment variables for flexibility
CITEWEAVE_DATA_DIR=${PWD}/data
CITEWEAVE_LOGS_DIR=${PWD}/logs
DOCS_WATCH_DIRS=${HOME}/Documents/research
```

## 🔒 Security Considerations

### Default Credentials
- **Neo4j**: `neo4j/12345678`
- **Qdrant**: No authentication by default
- **GROBID**: No authentication by default

### Production Deployment
1. Change default passwords
2. Use environment variables for secrets
3. Configure firewall rules
4. Enable HTTPS if needed

## 📊 Monitoring and Logs

### Query Traces
- Location: `./logs/query_traces.db`
- Format: SQLite database with 3 optimized tables
- Content: Complete agent workflow traces

### Application Logs
- Docker logs: `docker logs citeweave-app`
- File logs: `./data/watch_debug.log`

## 🚨 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Change port in .env
CITEWEAVE_API_PORT=8080
```

#### Permission Denied
```bash
# Ensure directories are writable
chmod 755 data logs documents
```

#### Docker Compose Errors
```bash
# Clean restart
./run.sh --rebuild
```

### Reset Configuration
```bash
# Remove all data and start fresh
rm -rf data logs documents .env
./scripts/setup_deployment.sh
./run.sh --rebuild
```

## 📚 Advanced Configuration

### Custom Document Sources
```bash
# Multiple document directories
DOCS_WATCH_DIRS=./documents,./papers,./research

# External directories (use absolute paths carefully)
DOCS_WATCH_DIRS=./documents,/mnt/shared/research
```

### Database Configuration
```bash
# Custom Neo4j settings
NEO4J_URI=bolt://my-neo4j-server:7687
NEO4J_USERNAME=myuser
NEO4J_PASSWORD=mypassword

# Custom Qdrant settings
QDRANT_URL=http://my-qdrant-server:6333
```

### Performance Tuning
```bash
# Memory limits
CITEWEAVE_MEMORY_LIMIT=4g

# Worker processes
CITEWEAVE_WORKERS=4
```

## 🔄 Updates and Maintenance

### Updating CiteWeave
```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
./run.sh --rebuild
```

### Backup and Restore
```bash
# Backup data
tar -czf citeweave-backup-$(date +%Y%m%d).tar.gz data/ logs/

# Restore data
tar -xzf citeweave-backup-20231201.tar.gz
```

## 📞 Support

For deployment issues:
1. Check the troubleshooting section
2. Review Docker logs: `docker logs citeweave-app`
3. Check environment configuration in `.env`
4. Ensure all required directories exist and are writable

---

**Note**: This deployment guide ensures that CiteWeave can be deployed on any machine without hardcoded paths or usernames. All configuration is done through environment variables and relative paths.
