import json

import numpy as np
from omegaconf import ListConfig, OmegaConf

from scripts.run_fast_foundation_stereo_video import (
    disparity_to_depth_original_size,
    make_depth_preview_panel_rgb,
    to_jsonable,
)


def test_fast_foundation_disparity_to_depth_rescales_to_original_size():
    disp = np.full((2, 3), 4.0, dtype=np.float32)

    depth = disparity_to_depth_original_size(
        disp,
        original_hw=(4, 6),
        fx=100.0,
        baseline_m=0.2,
        scale=0.5,
        min_disp=1e-3,
        max_depth_m=10.0,
    )

    assert depth.shape == (4, 6)
    assert np.nanmedian(depth) == 2.5


def test_fast_foundation_preview_panel_keeps_rgb_and_depth_side_by_side():
    left = np.zeros((4, 6, 3), dtype=np.uint8)
    depth = np.full((4, 6), 1.0, dtype=np.float32)

    panel = make_depth_preview_panel_rgb(left, depth, max_depth_m=4.0)

    assert panel.shape == (4, 12, 3)
    assert panel.dtype == np.uint8


def test_to_jsonable_converts_nested_config_objects():
    class FakeConfig:
        def __init__(self, value):
            self.value = value

        def items(self):
            return self.value.items()

    value = {"dims": [FakeConfig({"x": 1})]}

    assert to_jsonable(value) == {"dims": [{"x": 1}]}


def test_to_jsonable_converts_omegaconf_lists():
    value = OmegaConf.create({"vit_size": [518, 518], "nested": [{"x": 1}]})

    assert to_jsonable(value) == {"vit_size": [518, 518], "nested": [{"x": 1}]}


def test_to_jsonable_converts_direct_omegaconf_list_config():
    value = {"hidden_dims": ListConfig([128])}

    converted = to_jsonable(value)

    assert converted == {"hidden_dims": [128]}
    assert type(converted["hidden_dims"]) is list
    json.dumps(converted)
