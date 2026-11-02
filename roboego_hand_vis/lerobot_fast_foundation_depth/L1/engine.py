"""Keep the original model/numerics; overlap CPU decode and output with CUDA."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time
from uuid import uuid4

from .compute import episode_inputs
from .pipeline import OrderedWriter, prefetch, stereo_frame_range


def _is_unsupported_grid_sample_batch(error):
    message = str(error).lower()
    return "cudnn_status_not_supported" in message or "non-contiguous input" in message


class DepthEngine:
    def __init__(self, options):
        import cv2
        import torch
        import yaml
        from roboego_hand_vis.fast_foundation_depth.cli import config_from_yaml
        from roboego_hand_vis.fast_foundation_depth import compute as backend

        self.cv2, self.torch, self.backend, self.options = cv2, torch, backend, options
        threads = options.get("cpu_threads", 2)
        cv2.setNumThreads(threads)
        torch.set_num_threads(threads)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for depth inference")
        self.device = torch.device("cuda:0")
        config_path = Path(options["config"])
        settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        paths = settings.setdefault("paths", {})
        for key, override in (("fast_foundation_root", options.get("model_root")), ("ckpt", options.get("ckpt"))):
            path = Path(override or paths[key]).expanduser()
            paths[key] = str(path.resolve() if override or path.is_absolute() else (config_path.parent / path).resolve())
        settings.setdefault("parameters", {}).update(start=options["start"], num_frames=options["num_frames"], overwrite=True)
        for key in ("save_vis", "preview_every_n_frames"):
            if options.get(key) is not None:
                settings["parameters"][key] = options[key]
        self.config = config_from_yaml(settings, manifest=Path(), output_dir=Path())
        c = self.config
        if c.scale != .5 or c.valid_iters < 1:
            raise ValueError("High-throughput mode requires scale=0.5 and valid_iters >= 1")
        video_mode = options.get("depth_format") == "hevc12"
        if video_mode and not 0 < c.max_depth_m <= 4.095:
            raise ValueError("HEVC12 millimeter depth requires max_depth_m <= 4.095")
        if c.preview_every_n_frames < 1 or c.max_depth_m <= 0 or not (video_mode or c.save_depth_png or c.save_npy):
            raise ValueError("Invalid output/preview configuration")
        if not c.ckpt.is_file() or not c.ckpt.with_name("cfg.yaml").is_file() or not c.fast_foundation_root.is_dir():
            raise ValueError("Model source, checkpoint and adjacent cfg.yaml must exist")
        self.model, self.padder_class, self.amp_dtype, self.model_args = backend.load_model(
            root=c.fast_foundation_root, ckpt_path=c.ckpt, valid_iters=c.valid_iters,
            max_disp=c.max_disp, device=self.device)
        from .model_compat import patch_model_config
        self.compatibility = patch_model_config(self.model, self.model_args)
        if self.compatibility:
            print(f"Checkpoint compatibility defaults: {self.compatibility}", flush=True)

    def frames(self, inputs, state):
        """Decode and resize on the producer thread; preserve raw frame order."""
        cv2, c = self.cv2, self.config
        captures = []
        try:
            for side in ("left", "right"):
                # CAP_PROP_N_THREADS is only accepted as an open-time parameter.
                params = []
                if hasattr(cv2, "CAP_PROP_N_THREADS"):
                    params = [cv2.CAP_PROP_N_THREADS, self.options.get("cpu_threads", 2)]
                cap = cv2.VideoCapture(inputs["manifest"]["videos"][side]["path"], cv2.CAP_FFMPEG, params)
                captures.append(cap)
                if not cap.isOpened():
                    raise ValueError(f"Cannot open rectified {side} video")
            count = int(captures[0].get(cv2.CAP_PROP_FRAME_COUNT))
            offset = self.options.get("right_frame_offset", 0)
            begin, end = stereo_frame_range(count, int(captures[1].get(cv2.CAP_PROP_FRAME_COUNT)),
                                            c.start, c.num_frames, offset)
            captures[0].set(cv2.CAP_PROP_POS_FRAMES, begin)
            captures[1].set(cv2.CAP_PROP_POS_FRAMES, begin + offset)
            state.update(count=end - begin, start=begin, fps=captures[0].get(cv2.CAP_PROP_FPS) or 30.)
            print(f"Stereo pairing: left[{begin}:{end}], right[{begin + offset}:{end + offset}]", flush=True)
            for index in range(begin, end):
                started = time.perf_counter()
                images = []
                for cap in captures:
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError(f"Cannot read stereo frame {index}")
                    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                left, right = images
                if right.shape[:2] != left.shape[:2]:
                    right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
                left_small = cv2.resize(left, None, fx=c.scale, fy=c.scale, interpolation=cv2.INTER_AREA)
                right_small = cv2.resize(right, None, fx=c.scale, fy=c.scale, interpolation=cv2.INTER_AREA)
                yield index, left, left_small, right_small, time.perf_counter() - started
        finally:
            for cap in captures:
                cap.release()

    def infer_batch(self, pairs):
        if not pairs:
            raise ValueError("At least one stereo pair is required")
        torch, c = self.torch, self.config
        # Keep float32 inputs, padding, AMP dtype and model arguments identical
        # to the original implementation. Do not silently change precision.
        images = [
            torch.stack([torch.from_numpy(pair[side]) for pair in pairs])
            .pin_memory().to(self.device, non_blocking=True).float().permute(0, 3, 1, 2)
            for side in (0, 1)
        ]
        padder = self.padder_class(images[0].shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(*images)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        try:
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=self.amp_dtype):
                if c.hiera:
                    disp = self.model.run_hierachical(img0, img1, iters=c.valid_iters, test_mode=True, small_ratio=.5)
                else:
                    disp = self.model.forward(img0, img1, iters=c.valid_iters, test_mode=True, optimize_build_volume="pytorch1")
        except RuntimeError as exc:
            if len(pairs) == 1 or not _is_unsupported_grid_sample_batch(exc):
                raise
            print(
                f"Batch inference of {len(pairs)} pairs hit an unsupported cuDNN grid_sample layout; "
                "retrying these pairs individually.",
                flush=True,
            )
            import numpy as np
            results = [self.infer_batch([pair]) for pair in pairs]
            arrays = [result[0][0] for result in results]
            return np.stack(arrays), sum(result[1] for result in results)
        disparity = padder.unpad(disp.float()).detach()
        end.record()
        arrays = disparity.cpu().numpy()[:, 0].clip(0, None)
        # The blocking CPU transfer has already completed both timing events.
        return arrays, begin.elapsed_time(end) / 1000.

    def infer(self, left, right):
        arrays, gpu_s = self.infer_batch([(left, right)])
        return arrays[0], gpu_s

    def process(self, task):
        import numpy as np
        from roboego_hand_vis.common.depth_io import depth_path_for_frame, depth_uint16_png_path_for_frame, save_depth_uint16_png

        session, episode = Path(task["session"]), Path(task["episode"])
        inputs = episode_inputs(session, episode, camera_names=self.options.get("camera_names"))
        from roboego_hand_vis.common.stereo_quality import require_stereo_quality, GUANGLUN_POLICY
        quality_policy = GUANGLUN_POLICY if (session/'meta/guanglun_source.json').is_file() else None
        require_stereo_quality(session, episode, inputs['manifest']['videos']['left']['path'],
                               inputs['manifest']['videos']['right']['path'], policy=quality_policy)
        output = inputs["output"]
        offset = self.options.get("right_frame_offset", 0)
        if offset:
            tag = f"m{abs(offset)}" if offset < 0 else f"p{offset}"
            output = output.with_name(f"{output.name}_right_offset_{tag}")
        if output.is_symlink() or not output.resolve().is_relative_to(session.resolve()):
            raise ValueError("Depth output escapes session")
        batch_size = self.options.get("batch_size", 1)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        output.parent.mkdir(parents=True, exist_ok=True)
        work = output.parent / f".{output.name}.partial-{uuid4().hex}"
        work.mkdir()
        c = replace(self.config, output_dir=work, manifest=inputs["source_manifest"])
        cv2, torch, backend = self.cv2, self.torch, self.backend
        state, preview, depth_video = {}, [None], [None]
        video_mode = self.options.get("depth_format") == "hevc12"
        stats = dict(decode_s=0., gpu_s=0., inference_and_transfer_s=0., save_s=0.)
        processed = []
        started = time.perf_counter()

        def save(index, rgb, disparity):
            tick = time.perf_counter()
            depth = backend.disparity_to_depth_original_size(disparity, original_hw=rgb.shape[:2],
                fx=inputs["manifest"]["intrinsics"]["left"]["fx"],
                baseline_m=inputs["manifest"]["extrinsics"]["baseline_m"], scale=c.scale,
                min_disp=c.min_disp, max_depth_m=c.max_depth_m)
            if video_mode:
                from .depth_video import HEVC12Writer
                if depth_video[0] is None:
                    depth_video[0] = HEVC12Writer(work / "depth_hevc12.mkv", size=(depth.shape[1], depth.shape[0]),
                        fps=state["fps"], max_depth_m=c.max_depth_m, ffmpeg=self.options.get("ffmpeg", "ffmpeg"),
                        preset=self.options.get("hevc12_preset", "ultrafast"), threads=self.options.get("encode_threads", 4))
                depth_video[0].write(depth)
            if not video_mode and c.save_npy:
                np.save(depth_path_for_frame(work, index), depth)
            if not video_mode and c.save_depth_png and not save_depth_uint16_png(depth_uint16_png_path_for_frame(work, index), depth, max_depth_m=c.max_depth_m):
                raise RuntimeError(f"Cannot save depth frame {index}")
            if c.save_vis and backend.should_write_preview_frame(index, start_frame=state["start"], every_n_frames=c.preview_every_n_frames):
                from .visualization import make_preview
                panel = make_preview(rgb, depth, near=self.options.get("vis_min_depth", .2),
                    far=self.options.get("vis_max_depth", 1.), mode=self.options.get("vis_mode", "linear"))
                if preview[0] is None:
                    (work / "vis").mkdir(exist_ok=True)
                    preview[0] = cv2.VideoWriter(str(work / "vis/depth_preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                        backend.preview_video_fps(source_fps=state["fps"], every_n_frames=c.preview_every_n_frames),
                        (panel.shape[1], panel.shape[0]))
                    if not preview[0].isOpened():
                        raise RuntimeError("Cannot create depth preview")
                preview[0].write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            stats["save_s"] += time.perf_counter() - tick

        writer = OrderedWriter(save, self.options.get("write_queue", 4))
        frames = prefetch(self.frames(inputs, state), self.options.get("prefetch", 4))
        try:
            from itertools import islice
            frame_iter = iter(frames)
            while batch := list(islice(frame_iter, batch_size)):
                stats["decode_s"] += sum(frame[4] for frame in batch)
                tick = time.perf_counter()
                disparities, gpu_s = self.infer_batch([(frame[2], frame[3]) for frame in batch])
                stats["gpu_s"] += gpu_s
                stats["inference_and_transfer_s"] += time.perf_counter() - tick
                if len(disparities) != len(batch):
                    raise RuntimeError(f"Model returned {len(disparities)} disparities for {len(batch)} stereo pairs")
                for frame, disp in zip(batch, disparities):
                    index, rgb = frame[0], frame[1]
                    writer.submit(index, rgb, disp)
                    processed.append(index)
                if len(processed) == len(batch) or (len(processed) - len(batch)) // 300 < len(processed) // 300:
                    print(f"{len(processed)}/{state['count']} frames", flush=True)
            writer.close()
        finally:
            frames.close()
            try:
                writer.close()
            finally:
                if preview[0] is not None:
                    preview[0].release()
                if depth_video[0] is not None:
                    depth_video[0].close()
        elapsed = time.perf_counter() - started
        shutil.copy2(inputs["timestamps"], work / "source_timestamps.parquet")
        report = dict(stage="lerobot_fast_foundation_depth", engine="throughput", source_episode=str(episode),
            source_manifest=str(inputs["source_manifest"]), camera_feature=inputs["timestamps"].stem,
            processed_frames=processed, start=state["start"], requested_start=c.start,
            right_frame_offset=offset, right_index_lookup="right_index = left_index + right_frame_offset",
            scale=c.scale, valid_iters=c.valid_iters, batch_size=batch_size,
            elapsed_s=elapsed, fps=len(processed) / elapsed, timings=stats, synchronization_checked=False,
            timestamp_lookup="source_timestamps.parquet row indexed by original frame number")
        report["depth_storage"] = (dict(file="depth_hevc12.mkv", codec="hevc", pixel_format="gray12le",
            lossless=True, unit="millimeter", invalid_value=0, max_depth_m=c.max_depth_m,
            width=depth_video[0].size[0], height=depth_video[0].size[1], frames=depth_video[0].count,
            fps=state["fps"], video_index_to_source_index=f"source_index = video_index + {state['start']}")
            if video_mode else dict(format="per_frame_files"))
        report["checkpoint_compatibility"] = getattr(self, "compatibility", {})
        report["preview"] = dict(enabled=c.save_vis, every_n_frames=c.preview_every_n_frames,
            min_depth_m=self.options.get("vis_min_depth", .2), max_depth_m=self.options.get("vis_max_depth", 1.),
            mode=self.options.get("vis_mode", "linear"))
        (work / "lerobot_depth.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        metadata = dict(backend="fast_foundation_stereo", manifest_path=str(inputs["source_manifest"]),
            **inputs["manifest"], depth_unit="millimeter", depth_encoding="hevc12_lossless" if video_mode else "uint16_png", invalid_value=0,
            scale=c.scale, valid_iters=c.valid_iters, max_depth_m=c.max_depth_m, model_args=backend.to_jsonable(self.model_args))
        (work / "fast_foundation_stereo_video_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        backup = output.with_name(f".{output.name}.previous-{uuid4().hex}")
        old = output.exists()
        if old:
            output.rename(backup)
        try:
            work.rename(output)
        except BaseException:
            if old:
                backup.rename(output)
            raise
        if old:
            if backup.is_symlink() or not backup.resolve().is_relative_to(session.resolve()):
                raise ValueError("Unsafe previous output path")
            try:
                shutil.rmtree(backup)
            except OSError:
                print(f"Previous output retained: {backup}", flush=True)
        return dict(episode=str(episode), output=str(output), frames=len(processed), elapsed_s=elapsed, fps=report["fps"])
