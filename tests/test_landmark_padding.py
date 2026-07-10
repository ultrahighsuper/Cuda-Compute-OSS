"""Regression tests for pooled landmark attention with multi-block padding (issue #141).

When the sequence is not a multiple of num_landmarks and the tail padding spans
more than one block, `_pooled_landmarks` used to divide the boundary block by the
wrong count, emit fully-padding blocks as spurious all-zero landmarks, and point
their causal positions at padding indices. These pin the corrected behaviour.

CPU-safe: skips cleanly when torch is not installed.
Run:  python tests/test_landmark_padding.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

if torch is not None:
    from attention.hybrid import _pooled_landmarks, landmark_global_attention


def _skip_if_no_torch():
    if torch is None:
        print("SKIP  torch not installed")
        return True
    return False


# seq values whose tail padding spans MORE than one block for num_landmarks=64
# (block*landmarks - seq >= block), i.e. the regime the old code mishandled.
_TRIGGER_SEQS = [65, 100, 130, 2050]


def test_pooled_landmark_equals_true_block_mean_and_masks_padding():
    if _skip_if_no_torch():
        return
    torch.manual_seed(0)
    num_landmarks, dim = 64, 8
    for seq in _TRIGGER_SEQS:
        k = torch.randn(1, 1, seq, dim)
        v = torch.randn(1, 1, seq, dim)
        kl, vl, pos, valid = _pooled_landmarks(k, v, num_landmarks=num_landmarks)
        landmarks = min(num_landmarks, seq)
        block = math.ceil(seq / landmarks)
        for i in range(landmarks):
            lo, hi = i * block, min((i + 1) * block, seq)
            real = hi - lo
            if real > 0:
                assert bool(valid[i]), f"seq={seq} block {i} has {real} real tokens but marked invalid"
                assert torch.allclose(kl[0, 0, i], k[0, 0, lo:hi].mean(0), atol=1e-5), (
                    f"seq={seq} block {i}: K landmark != mean of its real tokens")
                assert torch.allclose(vl[0, 0, i], v[0, 0, lo:hi].mean(0), atol=1e-5)
                assert int(pos[i]) == hi - 1 <= seq - 1, (
                    f"seq={seq} block {i}: position {int(pos[i])} not the last real index {hi - 1}")
            else:
                assert not bool(valid[i]), f"seq={seq} block {i} is all padding but marked valid"


def test_no_valid_landmark_is_all_zero():
    """A valid pooled landmark must summarize real tokens, never be a padding zero-vector."""
    if _skip_if_no_torch():
        return
    torch.manual_seed(1)
    for seq in _TRIGGER_SEQS:
        k = torch.randn(1, 1, seq, 8)
        v = torch.randn(1, 1, seq, 8)
        kl, _vl, _pos, valid = _pooled_landmarks(k, v, num_landmarks=64)
        zero_and_valid = ((kl.abs().sum(-1) == 0) & valid.view(1, 1, -1)).sum().item()
        assert zero_and_valid == 0, f"seq={seq}: {zero_and_valid} valid landmarks are all-zero"


def test_landmark_global_attention_finite_on_trigger_seqs():
    if _skip_if_no_torch():
        return
    torch.manual_seed(2)
    for seq in _TRIGGER_SEQS:
        q = torch.randn(1, 1, seq, 8)
        k = torch.randn(1, 1, seq, 8)
        v = torch.randn(1, 1, seq, 8)
        out = landmark_global_attention(q, k, v, num_landmarks=64, policy="pooled")
        assert out.shape == q.shape
        assert torch.isfinite(out).all(), f"seq={seq}: non-finite pooled attention output"


def test_one_landmark_per_token_matches_exact():
    """num_landmarks == seq (block=1, no padding) must equal exact attention."""
    if _skip_if_no_torch():
        return
    from attention.reference import exact_attention
    torch.manual_seed(3)
    q = torch.randn(1, 2, 12, 4)
    k = torch.randn(1, 2, 12, 4)
    v = torch.randn(1, 2, 12, 4)
    out = landmark_global_attention(q, k, v, num_landmarks=12, policy="pooled")
    assert torch.allclose(out, exact_attention(q, k, v), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    fns = [v for kk, v in sorted(globals().items()) if kk.startswith("test_")]
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
