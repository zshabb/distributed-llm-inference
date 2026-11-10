#!/usr/bin/env bash
# Pre-commit guard: model weights and datasets are downloaded by scripts, never committed.
set -euo pipefail
limit=1048576
status=0
for f in "$@"; do
  if [ "$(wc -c < "$f")" -gt "$limit" ]; then
    echo "file exceeds 1 MiB, refusing to commit: $f" >&2
    status=1
  fi
done
exit "$status"
