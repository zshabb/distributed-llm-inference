#!/usr/bin/env bash
# Mutation check: inject known faults into block_manager.py one at a time; the stateful property
# test must FAIL for each. Restores the file from Git afterwards (requires a clean working copy).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
git diff --quiet -- src/dli/kv/block_manager.py || { echo "block_manager.py has uncommitted changes; refusing to mutate it" >&2; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
F=src/dli/kv/block_manager.py
run_mutant() {
  local name="$1" old="$2" new="$3"
  python3 - "$F" "$old" "$new" <<'PY'
import sys
p, old, new = sys.argv[1:]
s = open(p).read()
assert s.count(old) == 1, f"pattern not unique/found: {old!r}"
open(p, "w").write(s.replace(old, new))
PY
  if uv run --frozen pytest -q -x tests/unit/kv/test_block_manager_stateful.py >/tmp/mut.log 2>&1; then
    echo "MUTANT SURVIVED: $name"
  else
    echo "mutant killed:   $name  ($(grep -m1 -oE 'AssertionError: [^(]{0,70}|KV overwritten[^\"]{0,30}|prefix hit returned wrong KV' /tmp/mut.log | head -1))"
  fi
  git checkout -q -- "$F"
}
run_mutant "no copy-on-write" \
  "cow = pos % bs != 0 and self._ref[seq.blocks[-1]] > 1" "cow = False"
run_mutant "blocks treated as computed at allocation" \
  "num_computed=n_cached, namespace=namespace" "num_computed=len(token_ids), namespace=namespace"
run_mutant "non-leaf blocks offered for eviction" \
  "        self._num_cached_idle += 1
        if entry.children == 0:
            self.policy.add_candidate(b)" \
  "        self._num_cached_idle += 1
        self.policy.add_candidate(b)"
run_mutant "last prompt block served from cache (no rule 3)" \
  "eligible = (len(token_ids) - 1) // self.block_size * self.block_size" \
  "eligible = len(token_ids) // self.block_size * self.block_size"
git diff --quiet -- "$F" && echo "source restored: clean"
