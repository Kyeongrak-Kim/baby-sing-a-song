#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:4173/; then
  exit 0
fi
if [ ! -d dist ]; then
  npm run build
fi
nohup npm run preview > /tmp/baby-preview.log 2>&1 &
echo $! > /tmp/baby-preview.pid
