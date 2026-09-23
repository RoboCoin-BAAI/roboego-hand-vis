from __future__ import annotations

try:
    from .compute import depth_uint16_mm_to_preview_bgr
except ImportError:
    from compute import depth_uint16_mm_to_preview_bgr

__all__ = ["depth_uint16_mm_to_preview_bgr"]
