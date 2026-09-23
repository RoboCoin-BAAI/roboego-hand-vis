from __future__ import annotations

try:
    from .compute import (
        main,
        parse_args,
        prepare_segment,
        prepare_stereo_unit,
    )
except ImportError:
    from compute import (
        main,
        parse_args,
        prepare_segment,
        prepare_stereo_unit,
    )

__all__ = ["main", "parse_args", "prepare_segment", "prepare_stereo_unit"]
