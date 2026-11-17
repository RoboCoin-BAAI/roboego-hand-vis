"""Compatibility import; implementation lives in ../base/feature_storage.py."""
import sys
from ..base import feature_storage as _implementation

sys.modules[__name__] = _implementation
