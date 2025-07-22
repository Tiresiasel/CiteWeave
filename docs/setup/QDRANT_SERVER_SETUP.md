# CiteWeave Services Setup Guide

This guide covers setting up both Qdrant (vector database) and GROBID (PDF processing) services for CiteWeave using Docker Compose.

## Prerequisites

- Docker installed and running
- Docker Compose installed
- Python 3.8+ with required dependencies

## Quick Start

### 1. Start All Services

The easiest way to start both services is using the provided script:

```bash
python scripts/start_services.py
```

This script will:
- Check Docker and Docker Compose availability
- Start both Qdrant and GROBID services
- Wait for services to be ready
- Display service status

### 2. Manual Docker Compose Setup

Alternatively, you can use docker-compose directly:

```bash
# Start services in detached mode
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

## Services Overview

### Qdrant Vector Database
- **Port**: 6333 (REST API), 6334 (gRPC API)
- **Purpose**: Stores vector embeddings for semantic search
- **Health Check**: `http://localhost:6333/health`

### GROBID PDF Processor
- **Port**: 8070 (API)
- **Purpose**: Extracts metadata and references from PDF files
- **Health Check**: `http://localhost:8070/api/isalive`

## Configuration Files

### docker-compose.yml
Main configuration file defining both services:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
      - ./qdrant_config:/qdrant/config
    # ... other settings

  grobid:
    image: lfoppiano/grobid:0.8.0
    ports:
      - "8070:8070"
    environment:
      - JAVA_OPTS=-Xmx4g
    # ... other settings
```

### qdrant_config/qdrant.yaml
Qdrant server configuration:

```yaml
storage:
  storage_path: /qdrant/storage

service:
  http_port: 6333
  grpc_port: 6334
  enable_tls: false

cluster:
  enabled: false

telemetry:
  enabled: false
```

### config/qdrant_config.json
Client-side configuration for Python applications:

```json
{
  "host": "localhost",
  "port": 6333,
  "collections": {
    "sentences": {
      "vector_size": 384,
      "distance": "Cosine"
    },
    "paragraphs": {
      "vector_size": 384,
      "distance": "Cosine"
    },
    "sections": {
      "vector_size": 384,
      "distance": "Cosine"
    },
    "citations": {
      "vector_size": 384,
      "distance": "Cosine"
    }
  }
}
```

## Management Commands

### Start Services
```bash
docker-compose up -d
```

### Stop Services
```bash
docker-compose down
```

### Restart Services
```bash
docker-compose restart
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f qdrant
docker-compose logs -f grobid
```

### Check Status
```bash
docker-compose ps
```

### Remove Everything (including data)
```bash
docker-compose down -v
```

## Health Checks

### Qdrant Health Check
```bash
curl http://localhost:6333/health
```

Expected response:
```json
{
  "title": "qdrant",
  "version": "1.7.0"
}
```

### GROBID Health Check
```bash
curl http://localhost:8070/api/isalive
```

Expected response:
```json
{
  "status": "ok"
}
```

## Troubleshooting

### Services Won't Start

1. **Check Docker status**:
   ```bash
   docker --version
   docker-compose --version
   ```

2. **Check port availability**:
   ```bash
   lsof -i :6333
   lsof -i :8070
   ```

3. **Check Docker daemon**:
   ```bash
   docker info
   ```

### Qdrant Connection Issues

1. **Check if Qdrant is running**:
   ```bash
   docker-compose ps qdrant
   ```

2. **Check Qdrant logs**:
   ```bash
   docker-compose logs qdrant
   ```

3. **Verify configuration**:
   ```bash
   curl http://localhost:6333/collections
   ```

### GROBID Connection Issues

1. **Check if GROBID is running**:
   ```bash
   docker-compose ps grobid
   ```

2. **Check GROBID logs**:
   ```bash
   docker-compose logs grobid
   ```

3. **Test GROBID API**:
   ```bash
   curl http://localhost:8070/api/isalive
   ```

### Memory Issues

If GROBID fails to start due to memory constraints:

1. **Increase Docker memory limit** in Docker Desktop settings
2. **Reduce GROBID memory usage** by modifying `JAVA_OPTS` in docker-compose.yml:
   ```yaml
   environment:
     - JAVA_OPTS=-Xmx2g  # Reduce from 4g to 2g
   ```

### Data Persistence

- **Qdrant data** is stored in a Docker volume (`qdrant_storage`)
- **To backup data**:
  ```bash
  docker run --rm -v citeweave_qdrant_storage:/data -v $(pwd):/backup alpine tar czf /backup/qdrant_backup.tar.gz -C /data .
  ```

- **To restore data**:
  ```bash
  docker run --rm -v citeweave_qdrant_storage:/data -v $(pwd):/backup alpine tar xzf /backup/qdrant_backup.tar.gz -C /data
  ```

## Migration from Local Qdrant

If you were previously using a local Qdrant instance:

1. **Stop local Qdrant** (if running)
2. **Start Docker services**:
   ```bash
   docker-compose up -d
   ```
3. **Run migration script** (if available):
   ```bash
   python scripts/setup_qdrant_server.py
   ```

## Performance Tuning

### Qdrant Performance
- **Memory**: Qdrant uses memory-mapped files, so ensure sufficient RAM
- **Storage**: Use SSD for better performance
- **Collections**: Consider sharding for large datasets

### GROBID Performance
- **Memory**: Adjust `JAVA_OPTS` based on available RAM
- **CPU**: GROBID is CPU-intensive, ensure sufficient cores
- **Batch Processing**: For large PDF collections, process in batches

## Security Considerations

### Development Environment
- Services are exposed on localhost only
- No authentication required (development setup)
- TLS disabled for simplicity

### Production Environment
- Enable TLS/SSL
- Add authentication
- Use reverse proxy
- Restrict network access
- Regular security updates

## Monitoring

### Basic Monitoring
```bash
# Check resource usage
docker stats

# Monitor logs
docker-compose logs -f --tail=100
```

### Health Monitoring Script
```bash
#!/bin/bash
# Check both services
curl -f http://localhost:6333/health && echo "Qdrant OK" || echo "Qdrant FAILED"
curl -f http://localhost:8070/api/isalive && echo "GROBID OK" || echo "GROBID FAILED"
```

## Next Steps

After setting up the services:

1. **Test the setup** with the document processor:
   ```bash
   python src/processing/pdf/document_processor.py
   ```

2. **Verify vector indexing** works correctly

3. **Test GROBID metadata extraction** with sample PDFs

4. **Configure your application** to use the new services

For more information, see the main README.md and other documentation files. 