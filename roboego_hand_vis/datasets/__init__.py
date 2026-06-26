"""Dataset adapters that normalize raw ego-stereo data into one manifest."""

from .manifest import resolve_dataset_manifest, write_manifest

__all__ = ["resolve_dataset_manifest", "write_manifest"]
