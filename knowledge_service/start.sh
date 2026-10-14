#!/bin/bash
# Starts the two programs of this container (like search_service/entrypoint.sh): a private Redis (127.0.0.1 only, memory + cache,
# nothing is saved to disk) and the service. If either one dies the container exits, so Render restarts it.
set -u

if [ -z "${REDIS_URL:-}" ]; then
  redis-server --bind 127.0.0.1 --port 6379 --save "" --appendonly no --maxmemory 48mb --maxmemory-policy allkeys-lru --loglevel warning &
  REDIS_PID=$!
  export REDIS_URL="redis://127.0.0.1:6379/0"
else
  REDIS_PID=""                                   # an external Redis was configured: do not start one
fi

uvicorn --factory knowledge_service.app:create_from_env --host 0.0.0.0 --port "${PORT:-10000}" --workers 1 &
APP_PID=$!

stop() { kill $APP_PID ${REDIS_PID:-} 2>/dev/null; wait; exit 0; }
trap stop TERM INT

wait -n
echo "a program of the knowledge service stopped; exiting so the platform restarts it" >&2
kill $APP_PID ${REDIS_PID:-} 2>/dev/null
exit 1
