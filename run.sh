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
  # Use python to parse JSON and write overlay to file (avoid jq dependency)
  python3 - <<'PY' "${API_BASE}" "${COMPOSE_OVERLAY}" "${update_flag}"
import json, sys, urllib.request
api, out, update = sys.argv[1], sys.argv[2], sys.argv[3].lower()=='true'
body = json.dumps({"update_watch_map": update}).encode()
req = urllib.request.Request(api + "/mounts/overlay", data=body, headers={"Content-Type":"application/json"}, method="POST")
with urllib.request.urlopen(req) as r:
    data = json.loads(r.read().decode())
if not data.get('success'):
    print("[run.sh] ERROR: overlay generation failed", file=sys.stderr)
    print(json.dumps(data, indent=2), file=sys.stderr)
    sys.exit(1)
overlay = data.get('overlay','')
mounts = data.get('mounts', [])
open(out, 'w').write(overlay + ("\n" if not overlay.endswith("\n") else ""))
print("[run.sh] Overlay written:", out)
print("[run.sh] Planned binds:")
for m in mounts:
    print(" - {} -> {}".format(m.get('host'), m.get('container')))
PY
}

start_compose() {
  echo "[run.sh] Starting services with overlay..."
  docker compose -f "$COMPOSE_MAIN" -f "$COMPOSE_OVERLAY" up -d
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

gen_overlay
start_compose
wait_health || true
show_mounts
trigger_scan

echo "[run.sh] Done. Open http://localhost:31415"

docker compose up -d --build citeweave
docker compose watch