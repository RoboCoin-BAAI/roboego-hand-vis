"""Compatibility import; implementation lives in ../base/throughput.py."""
import sys
from ..base import throughput as _implementation

sys.modules[__name__] = _implementation
