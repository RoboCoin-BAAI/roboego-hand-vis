"""Compatibility import; implementation lives in ../base/reuse.py."""
import sys
from ..base import reuse as _implementation

sys.modules[__name__] = _implementation
