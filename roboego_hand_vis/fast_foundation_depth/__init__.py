"""Fast-FoundationStereo depth stage package."""

__all__ = [
    "FastFoundationDepthConfig",
    "depth_to_vis",
    "disparity_to_depth_original_size",
    "make_depth_preview_panel_rgb",
    "run_fast_foundation_depth",
    "to_jsonable",
]


def __getattr__(name: str):
    if name in {
        "FastFoundationDepthConfig",
        "disparity_to_depth_original_size",
        "run_fast_foundation_depth",
        "to_jsonable",
    }:
        from . import compute

        return getattr(compute, name)
    if name in {"depth_to_vis", "make_depth_preview_panel_rgb"}:
        from . import visualization

        return getattr(visualization, name)
    raise AttributeError(name)
