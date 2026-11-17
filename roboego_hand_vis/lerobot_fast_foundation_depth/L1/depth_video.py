"""Compatibility import; implementation lives in ../base/depth_video.py."""
import sys
from ..base import depth_video as _implementation

sys.modules[__name__] = _implementation
