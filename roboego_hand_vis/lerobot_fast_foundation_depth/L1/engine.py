"""Compatibility import; implementation lives in ../base/engine.py."""
import sys
from ..base import engine as _implementation

sys.modules[__name__] = _implementation
