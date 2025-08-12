#!/usr/bin/env bash
set -euo pipefail

# Run CiteWeave with dynamic host-folder mounts generated from watch_map
# Usage: ./run.sh [--update-watch-map]
#   --update-watch-map  Update server watch_map to container paths when generating overlay

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_BASE="${API_BASE:-http://localhost:31415/api/v1}"
COMPOSE_MAIN="${COMPOSE_MAIN:-docker-compose.yml}"
COMPOSE_OVERLAY="${COMPOSE_OVERLAY:-docker-compose.mounts.yml}"

update_flag=false
if [[ "${1:-}" == "--update-watch-map" ]]; then
  update_flag=true
fi

echo "[run.sh] Project root: ${ROOT_DIR}"
cd "$ROOT_DIR"

gen_overlay() {
  echo "[run.sh] Generating docker overlay from watch_map (update=${update_flag})"
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
    '      - app_data:/app/data',
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

start_compose() {
  echo "[run.sh] Starting services with overlay..."
  # 允许用户在不同架构/compose 文件名下运行
  if [[ -f docker-compose.amd64.yml ]]; then
    docker compose -f docker-compose.amd64.yml -f "$COMPOSE_OVERLAY" up -d
  else
    docker compose -f "$COMPOSE_MAIN" -f "$COMPOSE_OVERLAY" up -d
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
start_compose
wait_health || true
show_mounts
trigger_scan

echo "[run.sh] Done. Open http://localhost:31415"

docker compose up -d --build citeweave
docker compose watch