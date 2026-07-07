"""Attention playground for local experimentation.

This package is intentionally small and standalone. It is not wired into the
main scorer yet; it exists so the hybrid local-exact + spectral-global idea can
be tested honestly on one GPU before it is folded into a larger benchmark
track.
"""
from .reference import exact_attention
from .hybrid import (
    hybrid_attention,
    local_window_attention,
    spectral_global_mix,
)

__all__ = [
    "exact_attention",
    "local_window_attention",
    "spectral_global_mix",
    "hybrid_attention",
]
