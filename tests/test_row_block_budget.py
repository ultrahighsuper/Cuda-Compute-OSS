"""CPU-only tests for subspace._row_block's device budget (issue #138).

A streamed row-block costs more than the rows it stages: each iteration also
allocates the GEMM output, which cannot alias its operands and is live at the
same time. `_row_block` must budget those too, or the block overshoots
`vram_fraction x free` -- up to 2x at M = N -- and can OOM. This mirrors the
accounting `matmul/gemm.py` adopted in #95. Pure arithmetic; no GPU needed.

Run:  python tests/test_row_block_budget.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import subspace


class _FakeBackend:
    def __init__(self, free_bytes: int):
        self._free = free_bytes

    def free_compute_bytes(self) -> int:
        return self._free


FREE = 64 * 1024**2
FRAC = 0.3
ITEM = 4  # fp32


def _budget(free=FREE, frac=FRAC):
    return int(free * frac)


def test_out_cols_shrinks_the_block():
    """Counting the output columns must never pick a larger block."""
    n, m = 4096, 4096
    bk = _FakeBackend(FREE)
    without = subspace._row_block(n, n, bk, ITEM, FRAC)
    with_out = subspace._row_block(n, n, bk, ITEM, FRAC, out_cols=m)
    assert with_out < without


def test_block_stays_within_budget_at_m_equals_n():
    """M = N is the exactness path; staged rows + GEMM output must both fit."""
    n, m = 4096, 4096
    blk = subspace._row_block(n, n, _FakeBackend(FREE), ITEM, FRAC, out_cols=m)
    actual = blk * (n + m) * ITEM          # staged (blk,n) + output (blk,m)
    assert actual <= _budget()


def test_block_stays_within_budget_at_default_m():
    n, m = 4096, 4096 // 8
    blk = subspace._row_block(n, n, _FakeBackend(FREE), ITEM, FRAC, out_cols=m)
    actual = blk * (n + m) * ITEM
    assert actual <= _budget()


def test_old_model_would_have_overshot():
    """Regression witness: the pre-fix block (input rows only) exceeded budget
    once the (blk, m) GEMM output is counted -- 2x at M = N."""
    n, m = 4096, 4096
    old_blk = subspace._row_block(n, n, _FakeBackend(FREE), ITEM, FRAC)  # no out_cols
    old_actual = old_blk * (n + m) * ITEM
    assert old_actual > _budget()
    assert old_actual / _budget() > 1.9      # ~2x at M = N


def test_fixed_bytes_is_taken_off_the_budget():
    """stream_gemm_left_t's (n, m) product does not scale with the block, so it
    is charged up front rather than per row."""
    n, m = 1024, 256
    bk = _FakeBackend(FREE)
    fixed = n * m * ITEM
    blk = subspace._row_block(n, n, bk, ITEM, FRAC, fixed_bytes=fixed)
    assert blk * n * ITEM + fixed <= _budget()
    # and it must be no larger than the unconstrained block
    assert blk <= subspace._row_block(n, n, bk, ITEM, FRAC)


def test_block_never_drops_below_one():
    """Even when the fixed cost swallows the whole budget, stream at least one
    row rather than returning 0 (which would make the loop spin forever)."""
    n, m = 4096, 4096
    blk = subspace._row_block(n, n, _FakeBackend(1024), ITEM, FRAC,
                              out_cols=m, fixed_bytes=10**9)
    assert blk >= 1


def test_defaults_preserve_previous_behavior():
    """out_cols=0, fixed_bytes=0 reproduces the original formula exactly."""
    n, cols = 512, 512
    bk = _FakeBackend(FREE)
    expected = min(n, max(1, _budget() // (cols * ITEM)))
    assert subspace._row_block(n, cols, bk, ITEM, FRAC) == expected


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
