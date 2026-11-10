"""Model-based property test of BlockManager (Hypothesis rule-based state machine).

Besides the manager structural invariants, a *shadow KV store* simulates what the runner writes:
slot ``s`` holds an identifier of the exact token prefix whose KV it contains. After every step:

* every computed position of every live sequence must still hold *that sequence's* prefix, which
  catches writes into shared blocks, broken copy-on-write, and reuse of a block still in use;
* every prefix hit must return blocks whose shadow content equals the new prompt's prefix, which
  catches wrong reuse (hash or index bugs) and hits on uncomputed blocks.

Run more examples with ``HYPOTHESIS_PROFILE=focused`` (see tests/conftest.py).
"""

from __future__ import annotations

from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, consumes, invariant, rule

from dli.kv import BlockManager, NoFreeBlocksError

BS = 3
# A few shared "system prompts" so prefixes collide often, plus a tiny vocabulary for suffixes.
PREFIXES = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 9, 9], [7] * 7, []]
tokens = st.lists(st.integers(0, 4), min_size=0, max_size=7)


class BlockManagerMachine(RuleBasedStateMachine):
    seqs = Bundle("seqs")

    def __init__(self) -> None:
        super().__init__()
        self.policy = "lru"
        self.bm = BlockManager(num_blocks=12, block_size=BS, watermark=0.0)
        self.tokens: dict[int, list[int]] = {}  # seq -> full token list
        self.kv: dict[int, tuple[int, ...]] = {}  # slot -> token prefix whose KV it holds
        self.next_id = 0

    # -------------------------------------------------------------- helpers
    def _write_computed(self, seq: int, start: int, end: int) -> None:
        toks = self.tokens[seq]
        for p, slot in zip(range(start, end), self.bm.slot_mapping(seq, start, end), strict=True):
            self.kv[slot] = tuple(toks[: p + 1])

    # -------------------------------------------------------------- rules
    @rule(target=seqs, prefix=st.sampled_from(PREFIXES), suffix=tokens, policy=st.booleans())
    def allocate(self, prefix: list[int], suffix: list[int], policy: bool) -> object:
        prompt = prefix + suffix or [0]
        sid = self.next_id
        self.next_id += 1
        try:
            alloc = self.bm.allocate(sid, prompt)
        except NoFreeBlocksError:
            return None  # the bundle accepts None; rules below skip it
        # Rule 3 (runner contract): at least one prompt token must be computed to produce logits.
        # Found missing by fault injection: content checks alone accept a fully cached prompt.
        assert alloc.num_cached_tokens < len(prompt), "whole prompt served from cache"
        for p, slot in zip(
            range(alloc.num_cached_tokens),
            self.bm.slot_mapping(sid, 0, alloc.num_cached_tokens),
            strict=True,
        ):
            assert self.kv.get(slot) == tuple(prompt[: p + 1]), "prefix hit returned wrong KV"
        self.tokens[sid] = list(prompt)
        return sid

    @rule(seq=seqs, frac=st.floats(0, 1))
    def compute(self, seq: int | None, frac: float) -> None:
        if seq is None or not self.bm.has_sequence(seq):
            return
        done, total = self.bm.num_computed_tokens(seq), self.bm.num_tokens(seq)
        target = done + round((total - done) * frac)
        self._write_computed(seq, done, target)
        self.bm.mark_computed(seq, target)

    @rule(seq=seqs, new=st.lists(st.integers(0, 4), min_size=1, max_size=4))
    def append(self, seq: int | None, new: list[int]) -> None:
        if seq is None or not self.bm.has_sequence(seq):
            return
        # The runner computes everything before appending decode tokens.
        done, total = self.bm.num_computed_tokens(seq), self.bm.num_tokens(seq)
        self._write_computed(seq, done, total)
        self.bm.mark_computed(seq, total)
        try:
            ops = self.bm.append_slots(seq, new)
        except NoFreeBlocksError:
            return
        for op in ops:  # apply copy-on-write exactly as the KV store would
            for off in range(BS):
                src = op.src * BS + off
                if src in self.kv:
                    self.kv[op.dst * BS + off] = self.kv[src]
        self.tokens[seq].extend(new)

    @rule(target=seqs, seq=seqs)
    def fork(self, seq: int | None) -> object:
        if seq is None or not self.bm.has_sequence(seq):
            return None
        child = self.next_id
        self.next_id += 1
        self.bm.fork(seq, child)
        self.tokens[child] = list(self.tokens[seq])
        return child

    @rule(seq=consumes(seqs))
    def free(self, seq: int | None) -> None:
        if seq is None or not self.bm.has_sequence(seq):
            return
        self.bm.free(seq)
        del self.tokens[seq]

    @rule()
    def reset_cache(self) -> None:
        self.bm.reset_prefix_cache()

    # -------------------------------------------------------------- invariants
    @invariant()
    def structural_invariants(self) -> None:
        self.bm.check_invariants()

    @invariant()
    def every_live_sequence_still_owns_its_computed_kv(self) -> None:
        for seq, toks in self.tokens.items():
            n = self.bm.num_computed_tokens(seq)
            for p, slot in zip(range(n), self.bm.slot_mapping(seq, 0, n), strict=True):
                assert self.kv.get(slot) == tuple(toks[: p + 1]), (
                    f"seq {seq} position {p}: KV overwritten (shared block written?)"
                )


class CostAwareMachine(BlockManagerMachine):
    def __init__(self) -> None:
        super().__init__()
        self.bm = BlockManager(num_blocks=12, block_size=BS, eviction_policy="cost", watermark=0.0)


TestBlockManagerLRU = BlockManagerMachine.TestCase
TestBlockManagerCostAware = CostAwareMachine.TestCase
