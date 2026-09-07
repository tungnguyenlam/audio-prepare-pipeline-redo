#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

port="${1:-8888}"
notebook_dir="${2:-$PWD}"
state_root="${JUPYTER_EXPOSE_STATE_DIR:-$PWD/.jupyter-expose}"

for command_name in jupyter cloudflared openssl curl lsof; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'error: required command not found: %s\n' "$command_name" >&2
    exit 1
  fi
done

case "$port" in
  ''|*[!0-9]*)
    printf 'error: port must be a number\n' >&2
    exit 1
    ;;
esac

if (( port < 1 || port > 65535 )); then
  printf 'error: port must be between 1 and 65535\n' >&2
  exit 1
fi

if [[ ! -d "$notebook_dir" ]]; then
  printf 'error: notebook directory does not exist: %s\n' "$notebook_dir" >&2
  exit 1
fi

occupied_pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
if [[ -n "$occupied_pids" ]]; then
  while IFS= read -r occupied_pid; do
    [[ -n "$occupied_pid" ]] && kill "$occupied_pid" 2>/dev/null || true
  done <<<"$occupied_pids"

  for _ in {1..10}; do
    if ! lsof -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  occupied_pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$occupied_pids" ]]; then
    while IFS= read -r occupied_pid; do
      [[ -n "$occupied_pid" ]] && kill -KILL "$occupied_pid" 2>/dev/null || true
    done <<<"$occupied_pids"
    sleep 1
  fi

  if lsof -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    printf 'error: could not free port %s\n' "$port" >&2
    exit 1
  fi
fi

mkdir -p "$state_root"
run_dir="$(mktemp -d "$state_root/run.XXXXXX")"
jupyter_log="$run_dir/jupyter.log"
cloudflared_log="$run_dir/cloudflared.log"
token="$(openssl rand -hex 24)"

cleanup_failed_start() {
  if [[ -f "$run_dir/cloudflared.pid" ]]; then
    kill "$(<"$run_dir/cloudflared.pid")" 2>/dev/null || true
  fi
  if [[ -f "$run_dir/jupyter.pid" ]]; then
    kill "$(<"$run_dir/jupyter.pid")" 2>/dev/null || true
  fi
}

JUPYTER_TOKEN="$token" nohup jupyter lab \
  --no-browser \
  --ip=127.0.0.1 \
  --port="$port" \
  --ServerApp.root_dir="$notebook_dir" \
  --ServerApp.allow_remote_access=True \
  --ServerApp.trust_xheaders=True \
  >"$jupyter_log" 2>&1 &
jupyter_pid=$!
printf '%s\n' "$jupyter_pid" >"$run_dir/jupyter.pid"

jupyter_ready=0
for _ in {1..30}; do
  if ! kill -0 "$jupyter_pid" 2>/dev/null; then
    printf 'error: JupyterLab failed to start; see %s\n' "$jupyter_log" >&2
    cleanup_failed_start
    exit 1
  fi
  if curl --silent --show-error --max-time 1 "http://127.0.0.1:$port/lab" >/dev/null 2>&1; then
    jupyter_ready=1
    break
  fi
  sleep 1
done

if (( jupyter_ready == 0 )); then
  printf 'error: JupyterLab did not become ready; see %s\n' "$jupyter_log" >&2
  cleanup_failed_start
  exit 1
fi

nohup cloudflared tunnel \
  --url "http://127.0.0.1:$port" \
  --no-autoupdate \
  >"$cloudflared_log" 2>&1 &
cloudflared_pid=$!
printf '%s\n' "$cloudflared_pid" >"$run_dir/cloudflared.pid"

domain=""
tunnel_registered=0
for _ in {1..45}; do
  if ! kill -0 "$cloudflared_pid" 2>/dev/null; then
    printf 'error: cloudflared failed to start; see %s\n' "$cloudflared_log" >&2
    cleanup_failed_start
    exit 1
  fi
  domain="$(grep -Eo 'https://[[:alnum:]-]+\.trycloudflare\.com' "$cloudflared_log" | head -n 1 || true)"
  if grep -q 'Registered tunnel connection' "$cloudflared_log"; then
    tunnel_registered=1
  fi
  if [[ -n "$domain" ]] && (( tunnel_registered == 1 )); then
    break
  fi
  sleep 1
done

if [[ -z "$domain" ]] || (( tunnel_registered == 0 )); then
  printf 'error: Cloudflare tunnel did not become ready; see %s\n' "$cloudflared_log" >&2
  cleanup_failed_start
  exit 1
fi

public_host="${domain#https://}"
host_header_ready=0
for _ in {1..10}; do
  http_status="$(curl --silent --output /dev/null --write-out '%{http_code}' \
    --max-time 2 --header "Host: $public_host" \
    "http://127.0.0.1:$port/lab?token=$token" || true)"
  case "$http_status" in
    200|302)
      host_header_ready=1
      break
      ;;
  esac
  sleep 1
done

if (( host_header_ready == 0 )); then
  printf 'error: Jupyter rejected the tunnel hostname (status %s); see %s and %s\n' \
    "${http_status:-unknown}" "$jupyter_log" "$cloudflared_log" >&2
  cleanup_failed_start
  exit 1
fi

printf 'domain=%s\ntoken=%s\nport=%s\n' "$domain" "$token" "$port"