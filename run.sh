#!/usr/bin/env bash
set -euo pipefail

# Run CiteWeave with dynamic host-folder mounts generated from watch_map
# Usage: ./run.sh [OPTION]
#   (no option)        Stop containers, rebuild and restart all services (DEFAULT: updates watch_map + rebuilds)
#   --no-update-watch-map  Skip updating server watch_map (use existing mount configuration)
#   --no-rebuild      Skip rebuilding containers (use existing images)
#   --watch           Enter watch mode after starting services

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_BASE="${API_BASE:-http://localhost:31415/api/v1}"
COMPOSE_MAIN="${COMPOSE_MAIN:-docker-compose.yml}"
COMPOSE_OVERLAY="${COMPOSE_OVERLAY:-docker-compose.mounts.yml}"

# Load environment variables if .env file exists
if [[ -f ".env" ]]; then
    echo "[run.sh] Loading environment from .env file"
    export $(grep -v '^#' .env | xargs)
fi

# Default to updating watch_map and rebuilding unless explicitly disabled
update_flag=true
rebuild_flag=true

if [[ "${1:-}" == "--no-update-watch-map" ]]; then
  update_flag=false
fi

if [[ "${1:-}" == "--no-rebuild" ]]; then
  rebuild_flag=false
fi

echo "[run.sh] Project root: ${ROOT_DIR}"
cd "$ROOT_DIR"

gen_overlay() {
  if [[ "$update_flag" == "true" ]]; then
    echo "[run.sh] Generating docker overlay from watch_map (DEFAULT: updating watch_map)"
  else
    echo "[run.sh] Generating docker overlay from watch_map (SKIPPING: using existing configuration)"
  fi
  # Fetch mounts from API, then locally compose an overlay that filters out invalid hosts
  python3 - <<'PY' "${API_BASE}" "${COMPOSE_OVERLAY}" "${update_flag}"
import json, os, sys, urllib.request
api, out, update = sys.argv[1], sys.argv[2], sys.argv[3].lower()=='true'
def post(url, payload):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode())
data = post(api + "/mounts/overlay", {"update_watch_map": update})
if not data.get('success'):
    print("[run.sh] ERROR: overlay generation failed", file=sys.stderr)
    print(json.dumps(data, indent=2), file=sys.stderr)
    sys.exit(1)
mounts = data.get('mounts', []) or []
valid = []
skipped = []
for m in mounts:
    host = (m.get('host') or '').strip()
    cont = (m.get('container') or '').strip()
    # host 必须是宿主机真实路径且存在；不能以 /data/host 开头
    if host.startswith('/data/host') or not host.startswith('/') or not os.path.exists(host):
        skipped.append((host, cont))
    else:
        valid.append((host, cont))
lines = [
    'services:',
    '  citeweave:',
    '    volumes:',
    # Note: app_data volume removed, using direct host mount for data persistence
]
for h, c in valid:
    lines.append(f'      - {h}:{c}:ro')
overlay = "\n".join(lines) + "\n"
open(out, 'w').write(overlay)
print("[run.sh] Overlay written:", out)
print("[run.sh] Planned binds:")
for h, c in valid:
    print(f" - {h} -> {c}")
if skipped:
    print("[run.sh] Skipped mounts (host not valid or not shared):")
    for h, c in skipped:
        print(f"   - {h} -> {c}")
PY
}

stop_containers() {
  echo "[run.sh] Stopping existing containers..."
  # 停止所有相关的容器
  if [[ -f docker-compose.amd64.yml ]]; then
    docker compose -f docker-compose.amd64.yml -f "$COMPOSE_OVERLAY" down --remove-orphans 2>/dev/null || true
    docker compose -f docker-compose.amd64.yml down --remove-orphans 2>/dev/null || true
  else
    docker compose -f "$COMPOSE_MAIN" -f "$COMPOSE_OVERLAY" down --remove-orphans 2>/dev/null || true
    docker compose -f "$COMPOSE_MAIN" down --remove-orphans 2>/dev/null || true
  fi
  
  # 强制停止可能还在运行的容器
  docker stop citeweave-app citeweave-qdrant citeweave-grobid citeweave-neo4j 2>/dev/null || true
  docker rm citeweave-app citeweave-qdrant citeweave-grobid citeweave-neo4j 2>/dev/null || true
  
  echo "[run.sh] Containers stopped and removed"
}

start_compose() {
  if [[ "$rebuild_flag" == "true" ]]; then
    echo "[run.sh] Starting services with overlay and rebuilding..."
    # 允许用户在不同架构/compose 文件名下运行
    if [[ -f docker-compose.amd64.yml ]]; then
      docker compose -f docker-compose.amd64.yml -f "$COMPOSE_OVERLAY" up -d --build
    else
      docker compose -f "$COMPOSE_MAIN" -f "$COMPOSE_OVERLAY" up -d --build
    fi
  else
    echo "[run.sh] Starting services with overlay (no rebuild)..."
    # 允许用户在不同架构/compose 文件名下运行
    if [[ -f docker-compose.amd64.yml ]]; then
      docker compose -f docker-compose.amd64.yml -f "$COMPOSE_OVERLAY" up -d
    else
      docker compose -f "$COMPOSE_MAIN" -f "$COMPOSE_OVERLAY" up -d
    fi
  fi
}

start_base_if_needed() {
  echo -n "[run.sh] Checking API availability"
  if curl -fsS "${API_BASE%/api/v1}/api/v1/health" >/dev/null 2>&1; then
    echo " OK"
    return 0
  fi
  echo " -> starting base compose first"
  docker compose -f "$COMPOSE_MAIN" up -d
  wait_health || true
}

wait_health() {
  echo -n "[run.sh] Waiting for API health"
  for i in $(seq 1 60); do
    if curl -fsS "${API_BASE%/api/v1}/api/v1/health" >/dev/null 2>&1; then
      echo " OK"; return 0
    fi
    echo -n "."; sleep 1
  done
  echo "\n[run.sh] API did not become healthy in time"; return 1
}

trigger_scan() {
  echo "[run.sh] Triggering immediate scan"
  curl -fsS -X POST "${API_BASE}/watch/scan-now" >/dev/null || true
}

show_mounts() {
  echo "[run.sh] Current container mounts:"
  docker inspect citeweave-app --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}' || true
}

start_base_if_needed
gen_overlay
stop_containers
start_compose
wait_health || true
show_mounts
trigger_scan

echo "[run.sh] Done. Open http://localhost:31415"

# 如果需要进入watch模式（默认已经重建了）
if [[ "${1:-}" == "--watch" ]]; then
  echo "[run.sh] Entering watch mode..."
  docker compose watch
fi