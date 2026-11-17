"""Compatibility import; implementation lives in ../base/visualization.py."""
import sys
from ..base import visualization as _implementation

sys.modules[__name__] = _implementation
