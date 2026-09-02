import json
import urllib.error
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from roboego_hand_vis.quality.qwen_frame_annotation import (
    build_qwen_frame_annotation_messages,
    build_data_inspection_placeholder_frames,
    call_qwen_frame_annotations,
    completed_batch_indices,
    extract_video_frames_from_windows,
    is_data_inspection_failed,
    load_checkpoint_frames,
    migrate_object_attribute_hand_side,
    parse_qwen_frame_annotation_response,
    read_prompt_markdown,
    recover_checkpoint_from_raw,
    write_batch_annotation,
    write_checkpoint,
    write_chunk_annotations,
    validate_frame_annotations,
    write_qwen_frame_annotation_output,
)
from scripts.quality import run_qwen_frame_annotation


def write_image(path: Path) -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :, 1] = 180
    cv2.imwrite(str(path), image)


def write_video(path: Path, frame_count: int = 20, fps: float = 1.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 24))
    assert writer.isOpened()
    for idx in range(frame_count):
        frame = np.full((24, 32, 3), idx * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def write_prompt(path: Path) -> None:
    path.write_text("```text\nsystem\n```\n\n```text\nuser\n```\n", encoding="utf-8")


def valid_payload() -> dict:
    return {
        "frames": [
            {
                "t": 0,
                "scene": "store aisle",
                "activity": "wearer stands near shelves",
                "hand_visible": False,
                "hands_visible": {"left": False, "right": False},
                "wearer_region": {
                    "visible": False,
                    "bbox": None,
                    "body_parts": [],
                    "confidence": 0.0,
                    "notes": "",
                },
                "hand_object_interaction": False,
                "interaction_objects": [],
                "interaction_object_attributes": [],
                "attention_objects": [],
                "task_context": True,
                "task_objects": ["shelves"],
                "privacy": True,
                "privacy_types": ["face"],
                "image_quality": "clear",
                "uncertainty": 0.1,
                "notes": "people visible",
            },
            {
                "t": 1,
                "scene": "store shelf",
                "activity": "wearer holds a product",
                "hand_visible": True,
                "hands_visible": {"left": False, "right": True},
                "wearer_region": {
                    "visible": True,
                    "bbox": [0.55, 0.45, 0.98, 0.95],
                    "body_parts": ["right hand", "right forearm"],
                    "confidence": 0.85,
                    "notes": "wearer's right hand and forearm occupy the lower-right image region",
                },
                "hand_object_interaction": True,
                "interaction_objects": ["product"],
                "interaction_object_attributes": [
                    {
                        "name": "product",
                        "hand_side": "right",
                        "color": "green and yellow",
                        "shape": "bottle",
                        "material": "glass",
                        "visual_description": "green and yellow glass bottle",
                        "confidence": 0.9,
                    }
                ],
                "attention_objects": ["product"],
                "task_context": True,
                "task_objects": ["shelves", "products"],
                "privacy": False,
                "privacy_types": [],
                "image_quality": "clear",
                "uncertainty": 0.1,
                "notes": "",
            },
        ]
    }


def test_read_prompt_markdown_extracts_system_and_user_prompts():
    system_prompt, user_prompt = read_prompt_markdown(Path("docs/prompts/qwen_frame_annotation_v4.md"))

    assert "ego-centric" in system_prompt
    assert "hand_side" in user_prompt
    assert "activity 必须复用 interaction_objects" in system_prompt
    assert "纵向范围以召回优先" in system_prompt
    assert "只在横向范围上收紧" in system_prompt
    assert "y2 应接近或等于 1.0" in system_prompt
    assert "最小外接矩形" not in system_prompt
    assert "object_id" not in user_prompt


def test_read_english_v5_prompt_keeps_v4_schema_and_roi_policy():
    system_prompt, user_prompt = read_prompt_markdown(Path("docs/prompts/qwen_frame_annotation_v5_en.md"))

    assert "egocentric" in system_prompt
    assert "strict JSON" in system_prompt
    assert "coarse ROI" in system_prompt
    assert "vertical range" in system_prompt
    assert "tighten only the horizontal range" in system_prompt
    assert "y2 should be close to or equal to 1.0" in system_prompt
    assert "minimum bounding rectangle" not in system_prompt
    assert "wearer_region" in user_prompt
    assert "interaction_object_attributes" in user_prompt
    assert "object_id" not in user_prompt


def test_qwen_frame_annotation_cli_defaults_to_english_v5_prompt(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_qwen_frame_annotation.py", "--video", "left.mp4", "--output_json", "out.json"])

    args = run_qwen_frame_annotation.parse_args()

    assert args.prompt == "docs/prompts/qwen_frame_annotation_v5_en.md"


def test_build_qwen_frame_annotation_messages_includes_timestamps_and_images(tmp_path: Path):
    image_path = tmp_path / "frame.jpg"
    write_image(image_path)

    messages = build_qwen_frame_annotation_messages(
        system_prompt="system",
        user_prompt="user",
        frame_paths=[image_path],
        timestamps_s=[3.0],
    )

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    content = messages[1]["content"]
    assert content[0]["text"] == "user"
    assert content[1]["text"] == "Frame timestamp t=3 seconds"
    assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_extract_video_frames_from_windows_preserves_absolute_timestamps(tmp_path: Path):
    video = tmp_path / "input.mp4"
    frames_dir = tmp_path / "frames"
    write_video(video, frame_count=20, fps=1.0)

    frame_paths, timestamps_s = extract_video_frames_from_windows(
        video_path=video,
        output_dir=frames_dir,
        windows=[
            {"start": 2.0, "end": 5.0, "route_hint": "pass_to_qwen"},
            {"start": 10.0, "end": 12.0, "route_hint": "pass_to_qwen_with_aux"},
        ],
        fps=1.0,
        max_width=32,
    )

    assert timestamps_s == [2.0, 3.0, 4.0, 10.0, 11.0]
    assert [path.name for path in frame_paths] == [
        "frame_t000002.jpg",
        "frame_t000003.jpg",
        "frame_t000004.jpg",
        "frame_t000010.jpg",
        "frame_t000011.jpg",
    ]


def test_call_qwen_frame_annotations_returns_http_error_body(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    error = urllib.error.HTTPError(
        url="https://example.test",
        code=400,
        msg="Bad Request",
        hdrs={},
        fp=None,
    )
    error.read = lambda: b'{"code":"InvalidParameter","message":"bad input"}'

    with patch("urllib.request.urlopen", side_effect=error):
        parsed = call_qwen_frame_annotations(messages=[], model="qwen-vl-plus")

    assert parsed["status"] == "http_error"
    assert parsed["http_status"] == 400
    assert "InvalidParameter" in parsed["raw_response"]


def test_call_qwen_frame_annotations_retries_download_timeout_and_records_usage(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    error = urllib.error.HTTPError(
        url="https://example.test",
        code=400,
        msg="Bad Request",
        hdrs={},
        fp=None,
    )
    error.read = lambda: b'{"error":{"message":"Download multimodal file timed out"}}'

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "id": "chatcmpl-test",
                    "request_id": "request-test",
                    "usage": {"input_tokens": 11, "output_tokens": 13, "total_tokens": 24},
                    "choices": [{"message": {"content": json.dumps(valid_payload())}}],
                }
            ).encode("utf-8")

    calls = [error, Response()]

    def fake_urlopen(req, timeout):
        result = calls.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        parsed = call_qwen_frame_annotations(
            messages=[],
            model="qwen-vl-plus",
            max_retries=2,
            retry_sleep_s=0,
        )

    assert parsed["status"] == "ok"
    assert parsed["retry_count"] == 1
    assert parsed["attempt_count"] == 2
    assert parsed["usage"] == {"input_tokens": 11, "output_tokens": 13, "total_tokens": 24}
    assert parsed["request_id"] == "request-test"


def test_call_qwen_frame_annotations_retries_socket_timeout(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps(valid_payload())}}]}).encode("utf-8")

    calls = [TimeoutError("The read operation timed out"), Response()]

    def fake_urlopen(req, timeout):
        result = calls.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        parsed = call_qwen_frame_annotations(
            messages=[],
            model="qwen-vl-plus",
            max_retries=1,
            retry_sleep_s=0,
        )

    assert parsed["status"] == "ok"
    assert parsed["retry_count"] == 1
    assert parsed["attempt_count"] == 2


def test_call_qwen_frame_annotations_retries_validation_error_with_feedback(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    bad_payload = valid_payload()
    bad_payload["frames"][1]["wearer_region"]["bbox"] = [0.9, 0.2, 0.8, 0.9]

    class Response:
        def __init__(self, payload: dict):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    responses = [
        Response({"choices": [{"message": {"content": json.dumps(bad_payload)}}]}),
        Response({"choices": [{"message": {"content": json.dumps(valid_payload())}}]}),
    ]
    request_bodies = []

    def fake_urlopen(req, timeout):
        request_bodies.append(json.loads(req.data.decode("utf-8")))
        return responses.pop(0)

    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": [{"type": "text", "text": "annotate frames"}]},
    ]

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        parsed = call_qwen_frame_annotations(
            messages=messages,
            model="qwen-vl-plus",
            max_retries=1,
            retry_sleep_s=0,
        )

    assert parsed["status"] == "ok"
    assert parsed["retry_count"] == 1
    assert parsed["attempt_count"] == 2
    assert len(request_bodies) == 2
    retry_user_content = request_bodies[1]["messages"][1]["content"]
    retry_feedback = retry_user_content[-1]["text"]
    assert "Previous response failed schema validation" in retry_feedback
    assert "wearer_region.bbox" in retry_feedback


def test_data_inspection_failed_placeholder_frames_are_valid():
    parsed = {
        "status": "http_error",
        "raw_response": '{"error":{"code":"data_inspection_failed"}}',
    }

    frames = build_data_inspection_placeholder_frames([140.0, 141.0], parsed)

    assert is_data_inspection_failed(parsed) is True
    assert frames[0]["t"] == 140
    assert frames[0]["privacy"] is True
    assert frames[0]["privacy_types"] == ["face", "computer_screen", "checkout_screen"]
    assert frames[0]["uncertainty"] == 1.0
    assert "data_inspection_failed" in frames[0]["notes"]
    validate_frame_annotations(frames)


def test_qwen_annotation_cli_records_failed_batch_and_continues(monkeypatch, tmp_path: Path):
    prompt = tmp_path / "prompt.md"
    output_json = tmp_path / "annotations.json"
    frame0 = tmp_path / "frame0.jpg"
    frame1 = tmp_path / "frame1.jpg"
    write_prompt(prompt)
    write_image(frame0)
    write_image(frame1)

    monkeypatch.setattr(
        run_qwen_frame_annotation,
        "extract_video_frames",
        lambda **kwargs: ([frame0, frame1], [0.0, 1.0]),
    )
    responses = [
        {"status": "timeout_error", "raw_response": "read timed out", "retry_count": 1, "attempt_count": 2},
        {"status": "ok", "frames": [valid_payload()["frames"][1]], "usage": {"total_tokens": 9}},
    ]

    def fake_call_qwen_frame_annotations(**kwargs):
        return responses.pop(0)

    monkeypatch.setattr(run_qwen_frame_annotation, "call_qwen_frame_annotations", fake_call_qwen_frame_annotations)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_qwen_frame_annotation.py",
            "--video",
            str(tmp_path / "left.mp4"),
            "--output_json",
            str(output_json),
            "--prompt",
            str(prompt),
            "--batch_size",
            "1",
            "--retry_sleep_s",
            "0",
        ],
    )

    run_qwen_frame_annotation.main()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    checkpoint = json.loads((tmp_path / "annotations.json.checkpoint.json").read_text(encoding="utf-8"))
    assert payload["status"] == "partial_failed"
    assert payload["failed_batch_count"] == 1
    assert payload["failed_batches"][0]["batch_idx"] == 0
    assert payload["failed_batches"][0]["time_range"] == {"start": 0.0, "end": 0.0}
    assert len(payload["frames"]) == 1
    assert checkpoint["failed_batches"] == payload["failed_batches"]
    assert checkpoint["completed_batches"] == [1]


def test_parse_qwen_frame_annotation_response_accepts_markdown_json():
    response = "```json\n" + json.dumps(valid_payload()) + "\n```"

    parsed = parse_qwen_frame_annotation_response(response)

    assert parsed["status"] == "ok"
    assert parsed["frames"][1]["hands_visible"] == {"left": False, "right": True}
    assert parsed["frames"][1]["wearer_region"]["bbox"] == [0.55, 0.45, 0.98, 0.95]


def test_parse_qwen_frame_annotation_response_normalizes_missing_wearer_region():
    payload = valid_payload()
    del payload["frames"][1]["wearer_region"]

    parsed = parse_qwen_frame_annotation_response(json.dumps(payload))

    assert parsed["status"] == "ok"
    assert parsed["frames"][1]["wearer_region"] == {
        "visible": False,
        "bbox": None,
        "body_parts": [],
        "confidence": 0.0,
        "notes": "normalized_missing_wearer_region",
    }


def test_parse_qwen_frame_annotation_response_repairs_hands_visible_confidence_typo():
    response = """```json
{
  "frames": [
    {
      "t": 1,
      "scene": "store shelf",
      "activity": "wearer holds a product",
      "hand_visible": true,
      "hands_visible": {
        "left": false,
        "right": true
        "confidence": 0.9
      },
      "hand_object_interaction": true,
      "interaction_objects": ["product"],
      "interaction_object_attributes": [
        {
          "name": "product",
          "hand_side": "right",
          "color": "green and yellow",
          "shape": "bottle",
          "material": "glass",
          "visual_description": "green and yellow glass bottle",
          "confidence": 0.9
        }
      ],
      "attention_objects": ["product"],
      "task_context": true,
      "task_objects": ["shelves", "products"],
      "privacy": false,
      "privacy_types": [],
      "image_quality": "clear",
      "uncertainty": 0.1,
      "notes": ""
    }
  ]
}
```"""

    parsed = parse_qwen_frame_annotation_response(response)

    assert parsed["status"] == "ok"
    assert parsed["frames"][0]["hands_visible"] == {"left": False, "right": True}


def test_parse_qwen_frame_annotation_response_repairs_qwen_json_line_comment():
    payload = json.dumps(valid_payload(), indent=2)
    payload = payload.replace(
        '"right": true',
        '"right": true\n        // Hands are visible as they grip the pallet jack handles',
        1,
    )

    parsed = parse_qwen_frame_annotation_response(f"```json\n{payload}\n```")

    assert parsed["status"] == "ok"


def test_parse_qwen_frame_annotation_response_repairs_qwen_assignment_noise_line():
    payload = json.dumps(valid_payload(), indent=2)
    payload = payload.replace(
        '"scene": "store aisle"',
        'tape="508",\n      "scene": "store aisle"',
        1,
    )

    parsed = parse_qwen_frame_annotation_response(f"```json\n{payload}\n```")

    assert parsed["status"] == "ok"


def test_parse_qwen_frame_annotation_response_repairs_pointing_without_target_interaction():
    payload = valid_payload()
    payload["frames"] = [
        {
            "t": 89,
            "scene": "store aisle",
            "activity": "wearer points with the right index finger",
            "hand_visible": True,
            "hands_visible": {"left": False, "right": True},
            "hand_object_interaction": True,
            "interaction_objects": [],
            "interaction_object_attributes": [],
            "attention_objects": [],
            "task_context": True,
            "task_objects": ["shelves", "products"],
            "privacy": True,
            "privacy_types": ["face"],
            "image_quality": "clear",
            "uncertainty": 0.1,
            "notes": "",
        }
    ]

    parsed = parse_qwen_frame_annotation_response(json.dumps(payload))

    assert parsed["status"] == "ok"
    assert parsed["frames"][0]["hand_object_interaction"] is False
    assert "normalized_pointing_without_target" in parsed["frames"][0]["notes"]


def test_parse_qwen_frame_annotation_response_repairs_false_interaction_with_objects():
    payload = valid_payload()
    payload["frames"] = [payload["frames"][1]]
    payload["frames"][0]["activity"] = "wearer holds a phone in the left hand"
    payload["frames"][0]["hand_object_interaction"] = False
    payload["frames"][0]["interaction_objects"] = ["phone"]
    payload["frames"][0]["interaction_object_attributes"] = [
        {
            "name": "phone",
            "hand_side": "left",
            "color": "black",
            "shape": "rectangular",
            "material": "plastic and glass",
            "visual_description": "black rectangular phone held in the wearer's left hand",
            "confidence": 0.95,
        }
    ]

    parsed = parse_qwen_frame_annotation_response(json.dumps(payload))

    assert parsed["status"] == "ok"
    assert parsed["frames"][0]["hand_object_interaction"] is True
    assert "normalized_interaction_flag_from_objects" in parsed["frames"][0]["notes"]


def test_parse_qwen_frame_annotation_response_repairs_hand_visible_from_hands_visible():
    payload = valid_payload()
    payload["frames"] = [
        {
            "t": 842,
            "scene": "shopping cart area",
            "activity": "wearer looks at a shopping cart containing a product",
            "hand_visible": True,
            "hands_visible": {"left": False, "right": False},
            "hand_object_interaction": False,
            "interaction_objects": [],
            "interaction_object_attributes": [],
            "attention_objects": ["shopping cart", "product"],
            "task_context": True,
            "task_objects": ["shopping cart", "product"],
            "privacy": True,
            "privacy_types": ["face"],
            "image_quality": "clear",
            "uncertainty": 0.1,
            "notes": "wearer is no longer holding the product",
        }
    ]

    parsed = parse_qwen_frame_annotation_response(json.dumps(payload))

    assert parsed["status"] == "ok"
    assert parsed["frames"][0]["hand_visible"] is False
    assert "normalized_hand_visible_from_hands_visible" in parsed["frames"][0]["notes"]


def test_parse_qwen_frame_annotation_response_repairs_no_hand_interaction_objects():
    payload = valid_payload()
    payload["frames"] = [
        {
            "t": 291,
            "scene": "store aisle",
            "activity": "wearer pushes a shopping cart",
            "hand_visible": False,
            "hands_visible": {"left": False, "right": False},
            "hand_object_interaction": False,
            "interaction_objects": ["shopping cart"],
            "interaction_object_attributes": [
                {
                    "name": "shopping cart",
                    "hand_side": "unknown",
                    "color": "black",
                    "shape": "metal frame with wheels",
                    "material": "metal",
                    "visual_description": "black metal shopping cart",
                    "confidence": 0.95,
                }
            ],
            "attention_objects": [],
            "task_context": True,
            "task_objects": ["products"],
            "privacy": False,
            "privacy_types": [],
            "image_quality": "clear",
            "uncertainty": 0.05,
            "notes": "",
        }
    ]

    parsed = parse_qwen_frame_annotation_response(json.dumps(payload))

    assert parsed["status"] == "ok"
    assert parsed["frames"][0]["interaction_objects"] == []
    assert parsed["frames"][0]["interaction_object_attributes"] == []
    assert "shopping cart" in parsed["frames"][0]["attention_objects"]
    assert "shopping cart" in parsed["frames"][0]["task_objects"]
    assert "normalized_no_hand_interaction_objects" in parsed["frames"][0]["notes"]


def test_migrate_object_attribute_hand_side_infers_from_description():
    payload = valid_payload()
    del payload["frames"][1]["interaction_object_attributes"][0]["hand_side"]

    migrate_object_attribute_hand_side(payload["frames"])

    assert payload["frames"][1]["interaction_object_attributes"][0]["hand_side"] == "right"


def test_infer_object_hand_side_uses_phone_and_product_descriptions():
    frame = valid_payload()["frames"][1]

    migrate_object_attribute_hand_side([frame])
    assert frame["interaction_object_attributes"][0]["hand_side"] == "right"


def test_validate_frame_annotations_rejects_no_hand_interaction():
    payload = valid_payload()
    payload["frames"][0]["hand_object_interaction"] = True

    with pytest.raises(ValueError, match="hand_visible=false"):
        validate_frame_annotations(payload["frames"])


def test_validate_frame_annotations_rejects_interaction_flag_without_objects():
    payload = valid_payload()
    payload["frames"][1]["interaction_objects"] = []
    payload["frames"][1]["interaction_object_attributes"] = []

    with pytest.raises(ValueError, match="hand_object_interaction=true requires interaction_objects"):
        validate_frame_annotations(payload["frames"])


def test_validate_frame_annotations_rejects_objects_without_interaction_flag():
    payload = valid_payload()
    payload["frames"][1]["hand_object_interaction"] = False

    with pytest.raises(ValueError, match="interaction_objects require hand_object_interaction=true"):
        validate_frame_annotations(payload["frames"])


def test_validate_frame_annotations_rejects_invalid_wearer_region_bbox():
    payload = valid_payload()
    payload["frames"][1]["wearer_region"]["bbox"] = [0.8, 0.2, 0.7, 0.9]

    with pytest.raises(ValueError, match="wearer_region.bbox"):
        validate_frame_annotations(payload["frames"])


def test_write_qwen_frame_annotation_output_adds_metadata(tmp_path: Path):
    output = tmp_path / "annotations.json"

    payload = write_qwen_frame_annotation_output(
        output_json=output,
        parsed_frames=valid_payload()["frames"],
        video_path=Path("video.mp4"),
        prompt_path=Path("prompt.md"),
        model="qwen-vl-plus",
        sampling_fps=1.0,
        usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )

    saved = json.loads(output.read_text())
    assert saved == payload
    assert payload["model"] == "qwen-vl-plus"
    assert payload["usage"] == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    assert payload["usage_summary"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "frame_count": 2,
        "annotated_duration_s": 1.0,
        "tokens_per_frame": 7.5,
        "tokens_per_second": 15.0,
    }
    assert payload["frames"][1]["interaction_object_attributes"][0]["shape"] == "bottle"


def test_checkpoint_records_completed_batches_and_frames(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.json"

    write_checkpoint(
        checkpoint_path=checkpoint,
        frames=valid_payload()["frames"],
        completed_batches=[0, 1],
        video_path=Path("video.mp4"),
        prompt_path=Path("prompt.md"),
        model="qwen-vl-plus",
        sampling_fps=1.0,
        usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )

    assert completed_batch_indices(checkpoint) == {0, 1}
    assert len(load_checkpoint_frames(checkpoint)) == 2
    checkpoint_payload = json.loads(checkpoint.read_text())
    assert checkpoint_payload["usage"]["total_tokens"] == 15
    assert checkpoint_payload["usage_summary"]["tokens_per_frame"] == 7.5


def test_load_checkpoint_frames_normalizes_legacy_frames(tmp_path: Path):
    checkpoint = tmp_path / "legacy.checkpoint.json"
    payload = {
        "frames": valid_payload()["frames"],
        "completed_batches": [0],
    }
    del payload["frames"][1]["wearer_region"]
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    frames = load_checkpoint_frames(checkpoint)

    assert frames[1]["wearer_region"]["bbox"] is None
    assert frames[1]["wearer_region"]["notes"] == "normalized_missing_wearer_region"


def test_recover_checkpoint_from_raw_parses_completed_batches(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "batch_0000.raw.txt").write_text(json.dumps(valid_payload()), encoding="utf-8")

    checkpoint = tmp_path / "checkpoint.json"
    recovered = recover_checkpoint_from_raw(
        raw_output_dir=raw_dir,
        checkpoint_path=checkpoint,
        video_path=Path("video.mp4"),
        prompt_path=Path("prompt.md"),
        model="qwen-vl-plus",
        sampling_fps=1.0,
    )

    assert recovered["completed_batches"] == [0]
    assert len(recovered["frames"]) == 2


def test_write_batch_annotation_writes_named_batch_file(tmp_path: Path):
    output = write_batch_annotation(
        batch_output_dir=tmp_path,
        batch_idx=3,
        frames=valid_payload()["frames"],
        video_path=Path("video.mp4"),
        prompt_path=Path("prompt.md"),
        model="qwen-vl-plus",
        sampling_fps=1.0,
    )

    assert output.name == "batch_0003_t000000_000001.json"
    saved = json.loads(output.read_text())
    assert saved["batch_idx"] == 3
    assert len(saved["frames"]) == 2


def test_write_chunk_annotations_groups_batches(tmp_path: Path):
    batch_dir = tmp_path / "batches"
    chunk_dir = tmp_path / "chunks"
    write_batch_annotation(
        batch_output_dir=batch_dir,
        batch_idx=0,
        frames=valid_payload()["frames"],
        video_path=Path("video.mp4"),
        prompt_path=Path("prompt.md"),
        model="qwen-vl-plus",
        sampling_fps=1.0,
    )
    payload = valid_payload()
    payload["frames"][0]["t"] = 10
    payload["frames"][1]["t"] = 11
    write_batch_annotation(
        batch_output_dir=batch_dir,
        batch_idx=1,
        frames=payload["frames"],
        video_path=Path("video.mp4"),
        prompt_path=Path("prompt.md"),
        model="qwen-vl-plus",
        sampling_fps=1.0,
    )

    outputs = write_chunk_annotations(
        batch_output_dir=batch_dir,
        chunk_output_dir=chunk_dir,
        batches_per_chunk=5,
    )

    assert len(outputs) == 1
    assert outputs[0].name == "chunk_0000_batches_0000_0001_t000000_000011.json"
    saved = json.loads(outputs[0].read_text())
    assert saved["batch_indices"] == [0, 1]
    assert len(saved["frames"]) == 4


def test_write_chunk_annotations_removes_stale_partial_chunks(tmp_path: Path):
    batch_dir = tmp_path / "batches"
    chunk_dir = tmp_path / "chunks"
    write_batch_annotation(
        batch_output_dir=batch_dir,
        batch_idx=0,
        frames=valid_payload()["frames"],
        video_path=Path("video.mp4"),
        prompt_path=Path("prompt.md"),
        model="qwen-vl-plus",
        sampling_fps=1.0,
    )
    write_chunk_annotations(
        batch_output_dir=batch_dir,
        chunk_output_dir=chunk_dir,
        batches_per_chunk=5,
    )
    write_batch_annotation(
        batch_output_dir=batch_dir,
        batch_idx=1,
        frames=valid_payload()["frames"],
        video_path=Path("video.mp4"),
        prompt_path=Path("prompt.md"),
        model="qwen-vl-plus",
        sampling_fps=1.0,
    )

    write_chunk_annotations(
        batch_output_dir=batch_dir,
        chunk_output_dir=chunk_dir,
        batches_per_chunk=5,
    )

    chunk_files = sorted(path.name for path in chunk_dir.glob("chunk_*.json"))
    assert chunk_files == ["chunk_0000_batches_0000_0001_t000000_000001.json"]
