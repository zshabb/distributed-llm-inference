#!/usr/bin/env bash
# Pull benchmark results from the GPU host without letting the host's copies replace committed evidence.
#   pull_results.sh <source-dir> <repo-root>
# <source-dir> is an rsync source ("host:path" over $RSYNC_RSH, or a local directory in tests).
#
# The host keeps every result directory it ever wrote, including files from earlier runs that were
# later renamed, regenerated or committed in a different form locally. A plain `rsync host:results/
# results/` therefore let a stale host copy silently overwrite committed reference results.
#
# Rules:
#   * Files tracked by git under benchmarks/results are reference evidence: never written here.
#     If the host's copy differs, it is saved under benchmarks/results/.pod-conflicts/ (git-ignored,
#     same relative path) and listed, so nothing unique on the host is lost and the owner decides.
#   * Every other file (new results, or untracked local files of a run still in progress) is pulled
#     as before, newest host copy wins.
#   * Nothing is ever deleted, locally or on the host.
set -euo pipefail

[[ $# -eq 2 ]] || { echo "usage: $0 <source-dir> <repo-root>" >&2; exit 64; }
src="${1%/}/"
repo="$2"
dest="$repo/benchmarks/results"
conflicts="$dest/.pod-conflicts"
mkdir -p "$dest"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Tracked result files, relative to benchmarks/results/.
git -C "$repo" ls-files -z -- benchmarks/results \
  | sed -z 's|^benchmarks/results/||' | tr '\0' '\n' > "$work/tracked"
# The same as anchored rsync exclude patterns (wildcard characters escaped).
sed 's/[][*?\\]/\\&/g; s|^|/|' "$work/tracked" > "$work/exclude"

# 1. Everything that is not committed evidence.
rsync -a --exclude '/.pod-conflicts/' --exclude-from="$work/exclude" "$src" "$dest/"

# 2. Committed files whose host copy has different contents (checksum, not time or size).
#    Files the host does not have are fine; rsync reports them as vanished (exit 23), ignored here.
rsync -rn --checksum --existing --itemize-changes --files-from="$work/tracked" "$src" "$dest/" \
  2>/dev/null | awk '/^>f/ { sub(/^[^ ]+ /, ""); print }' > "$work/differ" || true

if [[ -s "$work/differ" ]]; then
  rsync -a --files-from="$work/differ" "$src" "$conflicts/"
  echo "warning: $(wc -l < "$work/differ") committed result file(s) differ on the host; kept the" \
       "committed copies, saved the host's under benchmarks/results/.pod-conflicts/:" >&2
  sed 's/^/  /' "$work/differ" >&2
fi
