"""Compatibility import; implementation lives in ../base/pipeline.py."""
import sys
from ..base import pipeline as _implementation

sys.modules[__name__] = _implementation
