"""Compatibility import; implementation lives in ../base/video_storage.py."""
import sys
from ..base import video_storage as _implementation

sys.modules[__name__] = _implementation
