#!/bin/sh
# Starts SearXNG (private) in the background, waits until it answers, then runs the gatekeeper.
# If either program stops, this script stops, so the container exits and Render restarts it.
set -u

INTERNAL_PORT="${SEARXNG_INTERNAL_PORT:-8081}"
export SEARXNG_INTERNAL_PORT="$INTERNAL_PORT"
DIED_FLAG="/tmp/searxng_died"
rm -f "$DIED_FLAG"

# --- Tell SearXNG where to listen. Several names are set because the variable depends on the image
# --- version (older: uWSGI, newer: Granian). Unused ones are harmless.
export SEARXNG_SETTINGS_PATH="${SEARXNG_SETTINGS_PATH:-/etc/searxng/settings.yml}"
export SEARXNG_BIND_ADDRESS="127.0.0.1"
export SEARXNG_PORT="$INTERNAL_PORT"
export BIND_ADDRESS="127.0.0.1:$INTERNAL_PORT"
export GRANIAN_HOST="127.0.0.1"
export GRANIAN_PORT="$INTERNAL_PORT"
# --- Keep memory low: one worker, few threads.
export GRANIAN_WORKERS=1
export GRANIAN_BLOCKING_THREADS=2
export GRANIAN_RUNTIME_THREADS=1
export UWSGI_WORKERS=1
export UWSGI_THREADS=2

# --- Find the image's own start script (path differs between versions).
SX_ENTRY=""
for candidate in /usr/local/searxng/entrypoint.sh /usr/local/searxng/dockerfiles/docker-entrypoint.sh /sbin/entrypoint.sh; do
    if [ -f "$candidate" ]; then SX_ENTRY="$candidate"; break; fi
done
if [ -z "$SX_ENTRY" ]; then
    echo "entrypoint: cannot find the SearXNG start script in the image" >&2
    exit 1
fi

# --- Find Python (the gatekeeper and the wait loop use only its standard library).
PYTHON=""
for candidate in python3 python /usr/local/searxng/.venv/bin/python /usr/local/searxng/venv/bin/python /usr/bin/python3; do
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
done
if [ -z "$PYTHON" ]; then
    echo "entrypoint: no Python found in the image" >&2
    exit 1
fi

MAIN_PID=$$
PROXY_PID=""

# --- 1. SearXNG in the background. It runs inside a subshell: when it ends for ANY reason, the
# ---    subshell leaves a flag file and sends TERM to this script.
(
    "$SX_ENTRY"
    echo "entrypoint: SearXNG exited with code $?" >&2
    : > "$DIED_FLAG"
    kill -TERM "$MAIN_PID"
) &

# --- Stop everything on TERM/INT (Render sends TERM). exit 1 if it was SearXNG that died.
on_signal() {
    [ -n "$PROXY_PID" ] && kill -TERM "$PROXY_PID" 2>/dev/null
    if [ -e "$DIED_FLAG" ]; then exit 1; fi
    exit 0
}
trap on_signal TERM INT

# --- 2. Wait (up to 120 s) until SearXNG answers on 127.0.0.1:INTERNAL_PORT. Any HTTP answer counts.
waited=0
until "$PYTHON" -c "
import http.client, sys
try:
    c = http.client.HTTPConnection('127.0.0.1', int(sys.argv[1]), timeout=2)
    c.request('GET', '/healthz'); c.getresponse().read(64)
except Exception:
    sys.exit(1)
" "$INTERNAL_PORT" 2>/dev/null; do
    waited=$((waited + 1))
    if [ "$waited" -ge 120 ]; then
        echo "entrypoint: SearXNG did not answer within 120 s" >&2
        exit 1
    fi
    sleep 1
done
echo "entrypoint: SearXNG is up, starting the gatekeeper on port ${PORT:-10000}"

# --- 3. The gatekeeper. If SEARCH_PROXY_TOKEN is missing it exits at once and so do we.
"$PYTHON" /opt/elvarn/proxy.py &
PROXY_PID=$!
wait "$PROXY_PID"
code=$?
echo "entrypoint: gatekeeper exited with code $code" >&2
exit "$code"
