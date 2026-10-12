from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from roboego_hand_vis.stereo_rectification.cammodel import create_camera_model, validate_stereo_camchain
from roboego_hand_vis.stereo_rectification.compute import prepare_stereo_unit
from roboego_hand_vis.stereo_rectification.equidistant import EquidistantCameraModel


def camera():
    return dict(camera_model="pinhole", distortion_model="equidistant",
                intrinsics=[550, 550, 960, 540], distortion_coeffs=[0.1, 0, 0, 0])


@pytest.mark.parametrize("model", [None, "equidistant", "fisheye"])
def test_supported_calibration_works(model):
    assert isinstance(create_camera_model(camera(), model), EquidistantCameraModel)
    assert validate_stereo_camchain({"cam0": camera(), "cam1": camera()}, model) == "equidistant"


@pytest.mark.parametrize("name", ["ds", "double_sphere", "doublesphere"])
@pytest.mark.parametrize("override", [None, "equidistant"])
def test_double_sphere_cannot_be_overridden_as_equidistant(name, override):
    info = camera()
    info["camera_model"] = name
    # Even a malformed DS file with four parameters cannot pass as pinhole.
    with pytest.raises(ValueError, match="双球面.*尚未"):
        create_camera_model(info, override)


@pytest.mark.parametrize("changes", [
    {"camera_model": "omni"}, {"camera_model": None},
    {"distortion_model": "radtan"}, {"distortion_model": "none"},
    {"intrinsics": [1, 2, 3, 4, 5, 6]}, {"distortion_coeffs": []},
])
def test_unsupported_or_malformed_calibration_rejected(changes):
    info = camera()
    info.update(changes)
    with pytest.raises(ValueError):
        create_camera_model(info, "equidistant")


def test_explicit_double_sphere_rejected_until_implementation_ready():
    with pytest.raises(ValueError, match="双球面.*尚未"):
        create_camera_model(camera(), "double_sphere")


def test_right_camera_validated_too():
    left = camera()
    right = deepcopy(left)
    right["camera_model"] = "ds"
    with pytest.raises(ValueError, match="cam1"):
        validate_stereo_camchain({"cam0": left, "cam1": right}, "equidistant")


@pytest.mark.parametrize("dry_run", [False, True])
def test_validation_precedes_overwrite_and_work_directory_creation(tmp_path, dry_run):
    calibration = tmp_path / "calibrations"
    calibration.mkdir()
    left = camera()
    left["camera_model"] = "ds"
    (calibration / "camera-camchain.yaml").write_text(yaml.safe_dump({"cam0": left, "cam1": camera()}))
    videos = tmp_path / "videos"
    videos.mkdir()
    existing = videos / "left_rectified.mp4"
    existing.write_bytes(b"preserve this output")
    with pytest.raises(ValueError, match="双球面"):
        prepare_stereo_unit(unit_root=tmp_path, session_dir=tmp_path, roboegopipe_root=Path(),
            dataset_adapter="raw", model="equidistant", balance=0, width=1600, height=900,
            overwrite=True, dry_run=dry_run)
    assert existing.read_bytes() == b"preserve this output"
    assert not (tmp_path / "_prepare_stereo_work").exists()


def test_supported_session_dry_run_is_preserved(tmp_path):
    calibration = tmp_path / "calibrations"
    calibration.mkdir()
    (calibration / "camera-camchain.yaml").write_text(yaml.safe_dump({"cam0": camera(), "cam1": camera()}))
    result = prepare_stereo_unit(unit_root=tmp_path, session_dir=tmp_path, roboegopipe_root=Path(),
        dataset_adapter="raw", model="equidistant", balance=0, width=1600, height=900,
        overwrite=True, dry_run=True)
    assert result["status"] == "dry_run"
    assert not (tmp_path / "_prepare_stereo_work").exists()


def test_low_level_entry_rejects_ds_before_output(tmp_path, monkeypatch):
    import sys
    from roboego_hand_vis.stereo_rectification import rectify_video

    source = tmp_path / "session"
    calibration = source / "calibrations"
    calibration.mkdir(parents=True)
    info = camera()
    info["camera_model"] = "ds"
    (calibration / "camera-camchain.yaml").write_text(yaml.safe_dump({"cam0": info, "cam1": info}))
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["rectify_video.py", "--input", str(source), "--output", str(output)])
    with pytest.raises(ValueError, match="双球面"):
        rectify_video.main()
    assert not output.exists()
