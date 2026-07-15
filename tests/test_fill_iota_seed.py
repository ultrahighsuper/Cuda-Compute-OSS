"""CPU-only tests that the `iota` fill consumes its seed (issue #104).

Before the fix, `_fill_iota` used the seed as a single additive constant
`(i + j + seed) % 97`, so distinct seeds produced the same matrix up to a
global shift: a couple's A (seed s) and B (seed s+1) differed by the constant 1
everywhere, and successive couples were identical in structure — eval silently
benchmarked `A @ A` on one repeated input.

Run:  python tests/test_fill_iota_seed.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matmul import storage as matmul_storage
from strategy import storage as strategy_storage

_MODULES = (matmul_storage, strategy_storage)


def _iota(mod, n, seed):
    mat = np.empty((n, n), dtype=np.float32)
    mod._fill_iota(mat, seed)
    return mat


def test_iota_is_deterministic_per_seed():
    # Same seed must reproduce the same matrix (deterministic benchmarking).
    for mod in _MODULES:
        assert np.array_equal(_iota(mod, 64, 5), _iota(mod, 64, 5))


def test_iota_distinct_seeds_differ_nontrivially():
    # Different seeds must give statistically independent matrices, not the
    # same matrix shifted by a global constant (the #104 bug).
    for mod in _MODULES:
        a = _iota(mod, 128, 0)
        b = _iota(mod, 128, 1)
        assert not np.array_equal(a, b)
        # The old bug made (b - a) a single constant everywhere; assert the
        # difference is genuinely non-constant now.
        diff = (b - a) % 97
        assert np.unique(diff).size > 1


def test_iota_couple_A_and_B_are_not_degenerate():
    # eval builds a couple as A=seed+2i, B=seed+2i+1. A and B must not be a
    # trivial offset of one another (which collapsed A@B toward A@A).
    for mod in _MODULES:
        A = _iota(mod, 96, 2 * 0)      # pair 0, A
        B = _iota(mod, 96, 2 * 0 + 1)  # pair 0, B
        assert not np.array_equal(A, B)
        assert np.unique(((B - A) % 97)).size > 1


def test_iota_successive_couples_differ():
    # Couple 0 (seeds 0,1) and couple 1 (seeds 2,3) must not be identical.
    for mod in _MODULES:
        couple0_A = _iota(mod, 96, 0)
        couple1_A = _iota(mod, 96, 2)
        assert not np.array_equal(couple0_A, couple1_A)


def test_iota_values_stay_in_range():
    for mod in _MODULES:
        m = _iota(mod, 80, 7)
        assert m.min() >= 0 and m.max() < 97


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
