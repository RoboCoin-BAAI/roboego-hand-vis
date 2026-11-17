"""Compatibility import; implementation lives in ../base/model_compat.py."""
import sys
from ..base import model_compat as _implementation

sys.modules[__name__] = _implementation
