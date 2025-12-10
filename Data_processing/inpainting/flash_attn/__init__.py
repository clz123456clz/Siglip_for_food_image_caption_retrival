"""
Dummy flash_attn module for Florence-2 on CPU/ARM.

This module exists only to bypass Florence-2's import check.
It does NOT implement any real flash attention functionality.
"""

__all__ = []

def flash_attn_func(*args, **kwargs):
    raise RuntimeError("flash_attn is not available on this platform (dummy stub).")
