import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def process_batch(
    model,
    frames,
    conf,
    device,
    imgsz,
):
    if not frames:
        return 0, 0, 0, 0

    results = model.predict(
        source=frames,
        conf=conf,
        imgsz=imgsz,
        device=device,
        max_det=4,
        verbose=False,
    )

    sample_count = 0
    hand_count = 0
    left_count = 0
    right_count = 0

    for result in results:
        sample_count += 1

        boxes = result.boxes

        if len(boxes) == 0:
            continue

        # 至少检测到一只手
        hand_count += 1

        left_present = False
        right_present = False

        for box in boxes:
            cls_id = int(box.cls.item())
            cls_name = model.names[cls_id].lower()

            if cls_name == "left":
                left_present = True
            elif cls_name == "right":
                right_present = True

        if left_present:
            left_count += 1

        if right_present:
            right_count += 1

    return (
        sample_count,
        hand_count,
        left_count,
        right_count,
    )


def process_video(
    video_path,
    model_path,
    output_json=None,
    sample_every=5,
    conf=0.15,
    device="0",
    imgsz=640,
    batch_size=32,
):
    video_path = Path(video_path)

    print(f"Loading model: {model_path}")

    model = YOLO(model_path)

    print(f"Model classes: {model.names}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise RuntimeError(
            f"Invalid FPS: {fps}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration = total_frames / fps

    detection_fps = fps / sample_every

    print()
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.3f}")
    print(f"Frames: {total_frames}")
    print(f"Duration: {duration:.2f}s")
    print(f"Sample every: {sample_every} frames")
    print(
        f"Detection frequency: "
        f"{detection_fps:.2f} Hz"
    )
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch_size}")
    print(f"Conf threshold: {conf}")
    print()

    batch_frames = []

    total_samples = 0
    total_hand = 0
    total_left = 0
    total_right = 0

    frame_index = 0

    while True:
        ok = cap.grab()

        if not ok:
            break

        # 每 N 帧取一帧
        if frame_index % sample_every != 0:
            frame_index += 1
            continue

        ok, frame = cap.retrieve()

        if not ok:
            frame_index += 1
            continue

        batch_frames.append(frame)

        if len(batch_frames) >= batch_size:
            (
                n_samples,
                n_hand,
                n_left,
                n_right,
            ) = process_batch(
                model=model,
                frames=batch_frames,
                conf=conf,
                device=device,
                imgsz=imgsz,
            )

            total_samples += n_samples
            total_hand += n_hand
            total_left += n_left
            total_right += n_right

            batch_frames.clear()

        frame_index += 1

    # 最后一批
    if batch_frames:
        (
            n_samples,
            n_hand,
            n_left,
            n_right,
        ) = process_batch(
            model=model,
            frames=batch_frames,
            conf=conf,
            device=device,
            imgsz=imgsz,
        )

        total_samples += n_samples
        total_hand += n_hand
        total_left += n_left
        total_right += n_right

    cap.release()

    if total_samples > 0:
        hand_ratio = total_hand / total_samples
        left_ratio = total_left / total_samples
        right_ratio = total_right / total_samples
    else:
        hand_ratio = 0.0
        left_ratio = 0.0
        right_ratio = 0.0

    print("=" * 60)

    print(
        f"Hand presence: "
        f"{hand_ratio * 100:.2f}%"
    )

    print(
        f"Left hand presence: "
        f"{left_ratio * 100:.2f}%"
    )

    print(
        f"Right hand presence: "
        f"{right_ratio * 100:.2f}%"
    )

    print("=" * 60)

    output = {
        "video": str(video_path),
        "model": str(model_path),

        "fps": round(fps, 4),
        "duration": round(duration, 4),

        "sample_every": sample_every,
        "detection_fps": round(
            detection_fps,
            4,
        ),

        "sample_count": total_samples,

        "hand_presence_ratio": round(
            hand_ratio,
            4,
        ),

        "hand_presence_percent": round(
            hand_ratio * 100,
            2,
        ),

        "left_hand_presence_percent": round(
            left_ratio * 100,
            2,
        ),

        "right_hand_presence_percent": round(
            right_ratio * 100,
            2,
        ),
    }

    if output_json:
        output_json = Path(output_json)

        output_json.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_json,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                output,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"Saved: {output_json}"
        )

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
    )

    parser.add_argument(
        "--model",
        default="detector.pt",
    )

    parser.add_argument(
        "--output_json",
        default=None,
    )

    parser.add_argument(
        "--sample_every",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--device",
        default="0",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    process_video(
        video_path=args.video,
        model_path=args.model,
        output_json=args.output_json,
        sample_every=args.sample_every,
        conf=args.conf,
        device=args.device,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()