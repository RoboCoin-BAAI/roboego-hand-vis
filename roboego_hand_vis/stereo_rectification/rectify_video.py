"""
双目立体校正脚本

功能：
1. 从双球面模型或等距投影（鱼眼）模型图像变换到无畸变针孔模型
2. 使用标定外参进行立体校正
3. 输出立体校正后的视频
4. 保存校正参数到 YAML 文件

相机模型实现：
- Double Sphere (双球面模型，尚未完成，当前入口禁止运行)
  - 使用虚拟针孔 + cv2.stereoRectify()
  - 需要 --fov 参数指定虚拟针孔视场角
- Equidistant (等距投影 / OpenCV fisheye 模型)
  - 使用 cv2.fisheye.stereoRectify() + cv2.fisheye.initUndistortRectifyMap()
  - 不需要虚拟针孔，--fov 参数被忽略

使用方式：
    # 自动检测模型（从 camchain 文件读取）
    python scripts/rectify_stereo_video.py \
        --input data/baai_ego_task_001/session_20260630_202318 \
        --output data/baai_ego_task_001_rectified
    
    # 手动指定模型类型
    python scripts/rectify_stereo_video.py \
        --input data/baai_ego_evaluate_004/session_xxx \
        --output data/output \
        --model equidistant
"""

import argparse
import os
import shutil
import json
import subprocess
import numpy as np
import cv2
import yaml
from pathlib import Path

try:
    from roboego_hand_vis.common.video_paths import find_named_video
except ImportError:
    import sys

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from roboego_hand_vis.common.video_paths import find_named_video

try:
    from .cammodel import (
        DoubleSphereCameraModel,
        EquidistantCameraModel,
        create_camera_model,
        load_camchain,
        get_model_type_name,
        validate_stereo_camchain,
    )
except ImportError:
    from cammodel import (
        DoubleSphereCameraModel,
        EquidistantCameraModel,
        create_camera_model,
        load_camchain,
        get_model_type_name,
        validate_stereo_camchain,
    )


def parse_extrinsic(cam_info: dict) -> tuple:
    """
    解析外参矩阵 T_cn_cnm1
    返回旋转矩阵 R 和平移向量 t
    """
    T = np.array(cam_info['T_cn_cnm1'])
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t


def compute_virtual_pinhole_params(
    width: int, height: int, fov_deg: float = 120.0
) -> tuple:
    """
    计算虚拟针孔相机参数
    fov_deg: 水平视场角（度）
    """
    fx = (width / 2) / np.tan(np.radians(fov_deg / 2))
    fy = fx
    cx = width / 2
    cy = height / 2
    return fx, fy, cx, cy


# ============================================================
# Double Sphere 管线
# ============================================================

def generate_combined_remap_double_sphere(
    ds_model: DoubleSphereCameraModel,
    fx_pin: float, fy_pin: float, cx_pin: float, cy_pin: float,
    output_width: int, output_height: int,
    R_rect: np.ndarray, P_rect: np.ndarray
) -> tuple:
    """
    [Double Sphere 管线] 生成组合的 remap map: 双球面 -> 虚拟针孔 -> 立体校正
    """
    # Step 1: 虚拟针孔 -> 双球面 的映射
    map_x_ds, map_y_ds = ds_model.generate_pinhole_to_double_sphere_map(
        fx_pin, fy_pin, cx_pin, cy_pin, output_width, output_height
    )
    
    # Step 2: 立体校正映射 (使用虚拟针孔内参和无畸变参数)
    K_pin = np.array([
        [fx_pin, 0, cx_pin],
        [0, fy_pin, cy_pin],
        [0, 0, 1]
    ], dtype=np.float64)
    D_pin = np.zeros((5, 1), dtype=np.float64)
    
    map_x_rect, map_y_rect = cv2.initUndistortRectifyMap(
        K_pin, D_pin, R_rect, P_rect, (output_width, output_height), cv2.CV_32FC1
    )
    
    # Step 3: 组合两个映射
    map_x_combined = cv2.remap(
        map_x_ds, map_x_rect, map_y_rect, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=-1
    )
    map_y_combined = cv2.remap(
        map_y_ds, map_x_rect, map_y_rect, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=-1
    )
    
    return map_x_combined, map_y_combined


def stereo_rectify_pinhole(
    K1: np.ndarray, K2: np.ndarray,
    image_size: tuple,
    R: np.ndarray, t: np.ndarray,
    alpha: float = 0.0
) -> dict:
    """
    [Double Sphere 管线] 使用无畸变针孔模型进行立体校正
    """
    D1 = np.zeros((5, 1), dtype=np.float64)
    D2 = np.zeros((5, 1), dtype=np.float64)
    
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, t, alpha=alpha
    )
    
    return {
        'R1': R1, 'R2': R2,
        'P1': P1, 'P2': P2,
        'Q': Q,
        'validPixROI1': validPixROI1,
        'validPixROI2': validPixROI2
    }


def run_double_sphere_pipeline(
    camera_model_left, camera_model_right,
    R, t,
    output_width, output_height, fov_deg, alpha,
    input_dir, output_session_dir, fps
):
    """
    运行 Double Sphere 管线的立体校正
    
    流程：
    1. 计算虚拟针孔参数 (基于 --fov)
    2. cv2.stereoRectify(K_pin, D=0) 计算校正参数
    3. 生成组合映射: DS → Pinhole → Rectify
    4. 处理视频
    """
    print(f"\n{'='*60}")
    print(f"Double Sphere 管线")
    print(f"{'='*60}")
    
    # 1. 计算虚拟针孔参数
    fx_pin, fy_pin, cx_pin, cy_pin = compute_virtual_pinhole_params(
        output_width, output_height, fov_deg
    )
    print(f"\n虚拟针孔参数:")
    print(f"  分辨率: {output_width}x{output_height}")
    print(f"  fx={fx_pin:.2f}, fy={fy_pin:.2f}, cx={cx_pin:.2f}, cy={cy_pin:.2f}")
    print(f"  水平FOV: {fov_deg}°")
    
    K_pin_left = np.array([
        [fx_pin, 0, cx_pin],
        [0, fy_pin, cy_pin],
        [0, 0, 1]
    ], dtype=np.float64)
    K_pin_right = K_pin_left.copy()
    
    # 2. 立体校正
    print(f"\n计算立体校正参数 (cv2.stereoRectify)...")
    rect_params = stereo_rectify_pinhole(
        K_pin_left, K_pin_right,
        (output_width, output_height),
        R, t, alpha
    )
    print(f"  R1:\n{rect_params['R1']}")
    print(f"  R2:\n{rect_params['R2']}")
    print(f"  P1:\n{rect_params['P1']}")
    print(f"  P2:\n{rect_params['P2']}")
    
    # 3. 生成 remap map
    print(f"\n生成 remap map (DS → Pinhole → Rectify)...")
    map_x_left, map_y_left = generate_combined_remap_double_sphere(
        camera_model_left, fx_pin, fy_pin, cx_pin, cy_pin,
        output_width, output_height,
        rect_params['R1'], rect_params['P1']
    )
    map_x_right, map_y_right = generate_combined_remap_double_sphere(
        camera_model_right, fx_pin, fy_pin, cx_pin, cy_pin,
        output_width, output_height,
        rect_params['R2'], rect_params['P2']
    )
    
    # 4. 处理视频
    _process_videos(input_dir, output_session_dir,
                    map_x_left, map_y_left, map_x_right, map_y_right, fps)
    
    # 5. 保存参数
    _save_rectify_params_ds(
        output_session_dir / 'calibrations' / 'rectify_params.yaml',
        K_pin_left, K_pin_right, rect_params,
        (output_width, output_height), fov_deg
    )
    
    return rect_params


# ============================================================
# Equidistant (Fisheye) 管线
# ============================================================

def run_equidistant_pipeline(
    camera_model_left, camera_model_right,
    R, t,
    output_width, output_height, balance,
    input_dir, output_session_dir, fps
):
    """
    运行 Equidistant 管线的立体校正
    
    流程：
    1. cv2.fisheye.stereoRectify(K, D, ...) 计算校正参数（使用实际鱼眼参数）
    2. cv2.fisheye.initUndistortRectifyMap(K, D, R, P) 生成映射
    3. 处理视频
    
    注意：Equidistant 模型始终使用原始图像分辨率，--width/--height 被忽略。
    原因：与 Double Sphere 不同，Equidistant 是重映射而非裁剪虚拟针孔。
    如果 newImageSize 比原始分辨率小，OpenCV 会按比例缩小焦距，
    导致图像质量下降并产生黑边。
    """
    print(f"\n{'='*60}")
    print(f"Equidistant (Fisheye) 管线")
    print(f"{'='*60}")
    
    # 获取原始图像尺寸（Equidistant 始终使用原始分辨率）
    orig_resolution = None
    left_video = find_named_video(input_dir / 'videos', 'left')
    if left_video.exists():
        cap = cv2.VideoCapture(str(left_video))
        if cap.isOpened():
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            orig_resolution = (orig_w, orig_h)
            cap.release()
    
    if orig_resolution is None:
        raise RuntimeError(
            "Equidistant 管线需要读取原始视频获取分辨率。"
            f"无法打开视频: {left_video}"
        )
    
    orig_w, orig_h = orig_resolution
    print(f"\n原始图像尺寸: {orig_w}x{orig_h}")
    
    if (orig_w, orig_h) != (output_width, output_height):
        print(f"注意: Equidistant 管线始终使用原始分辨率 {orig_w}x{orig_h}，")
        print(f"      --width {output_width} --height {output_height} 被忽略。")
    
    # 1. 立体校正（使用 cv2.fisheye.stereoRectify）
    # image_size 和 newImageSize 都使用原始分辨率
    print(f"\n计算立体校正参数 (cv2.fisheye.stereoRectify)...")
    rect_params = EquidistantCameraModel.stereo_rectify(
        camera_model_left, camera_model_right,
        orig_resolution, R, t,
        new_image_size=orig_resolution, balance=balance
    )
    print(f"  R1:\n{rect_params['R1']}")
    print(f"  R2:\n{rect_params['R2']}")
    print(f"  P1:\n{rect_params['P1']}")
    print(f"  P2:\n{rect_params['P2']}")
    
    # 2. 生成 remap map（使用 cv2.fisheye.initUndistortRectifyMap）
    print(f"\n生成 remap map (cv2.fisheye.initUndistortRectifyMap)...")
    map_x_left, map_y_left = camera_model_left.generate_stereo_remap(
        rect_params['R1'], rect_params['P1'],
        orig_w, orig_h
    )
    map_x_right, map_y_right = camera_model_right.generate_stereo_remap(
        rect_params['R2'], rect_params['P2'],
        orig_w, orig_h
    )
    
    # 3. 自动裁剪黑边
    print(f"\n自动裁剪黑边...")
    roi_left = compute_valid_crop_roi(map_x_left, map_y_left, orig_w, orig_h)
    roi_right = compute_valid_crop_roi(map_x_right, map_y_right, orig_w, orig_h)
    roi = intersect_roi(roi_left, roi_right)
    
    crop_w = roi[2] - roi[0]
    crop_h = roi[3] - roi[1]
    print(f"  左图 ROI:  x=[{roi_left[0]}, {roi_left[2]}) y=[{roi_left[1]}, {roi_left[3]})  ({roi_left[2]-roi_left[0]}x{roi_left[3]-roi_left[1]})")
    print(f"  右图 ROI:  x=[{roi_right[0]}, {roi_right[2]}) y=[{roi_right[1]}, {roi_right[3]})  ({roi_right[2]-roi_right[0]}x{roi_right[3]-roi_right[1]})")
    print(f"  交集 ROI:  x=[{roi[0]}, {roi[2]}) y=[{roi[1]}, {roi[3]})  → {crop_w}x{crop_h}")
    
    x_min, y_min = roi[0], roi[1]
    
    # 确保裁剪尺寸为偶数（H.264 yuv420p 要求宽高均为偶数）
    if crop_w % 2 != 0:
        crop_w -= 1
        roi = (roi[0], roi[1], roi[0] + crop_w, roi[3])
    if crop_h % 2 != 0:
        crop_h -= 1
        roi = (roi[0], roi[1], roi[2], roi[1] + crop_h)
    
    if crop_w < orig_w or crop_h < orig_h:
        map_x_left, map_y_left = crop_remap_maps(map_x_left, map_y_left, roi)
        map_x_right, map_y_right = crop_remap_maps(map_x_right, map_y_right, roi)
        print(f"  已裁剪: {orig_w}x{orig_h} → {crop_w}x{crop_h}")
        
        # 裁剪后主点偏移，更新 P1, P2, Q
        # cx_new = cx_old - x_min, cy_new = cy_old - y_min
        print(f"  更新投影矩阵主点偏移: dx=-{x_min}, dy=-{y_min}")
        rect_params['P1'] = rect_params['P1'].copy()
        rect_params['P2'] = rect_params['P2'].copy()
        rect_params['P1'][0, 2] -= x_min  # cx
        rect_params['P1'][1, 2] -= y_min  # cy
        rect_params['P2'][0, 2] -= x_min  # cx
        rect_params['P2'][1, 2] -= y_min  # cy
        
        # 更新 Q 矩阵
        # Q[0,3] = -cx, Q[1,3] = -cy (for left camera)
        # Q[2,3] = f (focal length, unchanged)
        rect_params['Q'] = rect_params['Q'].copy()
        rect_params['Q'][0, 3] += x_min   # -cx_new = -(cx_old - x_min) = -cx_old + x_min
        rect_params['Q'][1, 3] += y_min   # -cy_new = -(cy_old - y_min) = -cy_old + y_min
    else:
        print(f"  无需裁剪")
    
    # 4. 处理视频
    _process_videos(input_dir, output_session_dir,
                    map_x_left, map_y_left, map_x_right, map_y_right, fps)
    
    # 5. 保存参数（记录裁剪后的尺寸和更新后的 P/Q）
    _save_rectify_params_equidistant(
        output_session_dir / 'calibrations' / 'rectify_params.yaml',
        camera_model_left, camera_model_right, rect_params,
        (crop_w, crop_h),
        crop_roi=roi
    )
    
    return rect_params, (crop_w, crop_h)


# ============================================================
# 有效区域裁剪（消除黑边）
# ============================================================

def compute_valid_crop_roi(
    map_x: np.ndarray, map_y: np.ndarray,
    src_width: int, src_height: int,
    valid_ratio_threshold: float = 0.95
) -> tuple:
    """
    分析 remap map，找出无黑边的最大矩形区域。

    原理：remap 映射到原始图像范围外的像素会因 BORDER_CONSTANT 而变为黑色。
    通过统计每行/每列的有效像素比例，找出连续有效比例超过阈值的范围。

    Args:
        map_x, map_y: remap 映射表 (output_h, output_w)
        src_width, src_height: 原始输入图像尺寸
        valid_ratio_threshold: 行/列有效像素比例阈值 (默认 0.95)

    Returns:
        (x_min, y_min, x_max, y_max): 裁剪区域坐标
    """
    out_h, out_w = map_x.shape

    # 有效掩码：映射坐标在原始图像范围内（留 2px margin 避免半像素插值边缘）
    margin = 2
    valid_mask = (
        (map_x >= margin) & (map_x < src_width - margin) &
        (map_y >= margin) & (map_y < src_height - margin)
    )

    # 对每行统计有效比例
    row_valid_ratio = np.sum(valid_mask, axis=1) / out_w
    # 对每列统计有效比例
    col_valid_ratio = np.sum(valid_mask, axis=0) / out_h

    # 找出行方向上有效比例 >= 阈值的连续范围
    valid_rows = np.where(row_valid_ratio >= valid_ratio_threshold)[0]
    valid_cols = np.where(col_valid_ratio >= valid_ratio_threshold)[0]

    if len(valid_rows) == 0 or len(valid_cols) == 0:
        # 如果无法检测，返回完整区域
        return (0, 0, out_w, out_h)

    y_min = int(valid_rows[0])
    y_max = int(valid_rows[-1]) + 1
    x_min = int(valid_cols[0])
    x_max = int(valid_cols[-1]) + 1

    return (x_min, y_min, x_max, y_max)


def intersect_roi(roi1: tuple, roi2: tuple) -> tuple:
    """
    计算两个 ROI 的交集（取较严格的边界）。

    Args:
        roi1, roi2: (x_min, y_min, x_max, y_max)

    Returns:
        交集 ROI
    """
    return (
        max(roi1[0], roi2[0]),
        max(roi1[1], roi2[1]),
        min(roi1[2], roi2[2]),
        min(roi1[3], roi2[3]),
    )


def crop_remap_maps(map_x, map_y, roi):
    """
    根据 ROI 裁剪 remap maps。

    Args:
        map_x, map_y: 原始 remap 映射表
        roi: (x_min, y_min, x_max, y_max)

    Returns:
        裁剪后的 (map_x, map_y)
    """
    x_min, y_min, x_max, y_max = roi
    return map_x[y_min:y_max, x_min:x_max], map_y[y_min:y_max, x_min:x_max]


# ============================================================
# 公共工具函数
# ============================================================

def process_video(
    input_path: str, output_path: str,
    map_x: np.ndarray, map_y: np.ndarray,
    fps: float
) -> int:
    """
    处理视频：应用 remap 并通过 ffmpeg 管道输出 H.264 编码视频。
    返回处理的帧数。

    优先使用 ffmpeg (libx264) 编码，兼容性好，可在 VS Code / 浏览器中播放。
    如果 ffmpeg 不可用，回退到 cv2.VideoWriter + mp4v。
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {input_path}")

    out_height, out_width = map_x.shape
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 尝试使用 ffmpeg 管道
    ffmpeg_proc = None
    try:
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{out_width}x{out_height}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            '-crf', '18',
            '-movflags', '+faststart',
            output_path
        ]
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        use_ffmpeg = True
    except FileNotFoundError:
        use_ffmpeg = False
        print("  ffmpeg 不可用，回退到 cv2.VideoWriter (mp4v)")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rectified_frame = cv2.remap(
            frame, map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

        if use_ffmpeg:
            ffmpeg_proc.stdin.write(rectified_frame.tobytes())
        else:
            writer.write(rectified_frame)
        frame_count += 1

    cap.release()
    if use_ffmpeg:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
    else:
        writer.release()

    return frame_count


def _process_videos(input_dir, output_session_dir,
                    map_x_left, map_y_left, map_x_right, map_y_right, fps):
    """处理左右视频"""
    # 创建输出目录
    os.makedirs(output_session_dir / 'videos', exist_ok=True)
    
    # 处理 left 视频
    left_input = find_named_video(input_dir / 'videos', 'left')
    left_output = output_session_dir / 'videos' / 'left_rectified.mp4'
    print(f"\n处理左相机视频: {left_input} -> {left_output}")
    left_frames = process_video(str(left_input), str(left_output),
                                map_x_left, map_y_left, fps)
    print(f"  处理帧数: {left_frames}")
    
    # 处理 right 视频
    right_input = find_named_video(input_dir / 'videos', 'right')
    right_output = output_session_dir / 'videos' / 'right_rectified.mp4'
    print(f"\n处理右相机视频: {right_input} -> {right_output}")
    right_frames = process_video(str(right_input), str(right_output),
                                 map_x_right, map_y_right, fps)
    print(f"  处理帧数: {right_frames}")


def _save_rectify_params_ds(
    output_path, K_pin_left, K_pin_right, rect_params, image_size, fov_deg
):
    """保存 Double Sphere 管线的校正参数"""
    K1_rect = rect_params['P1'][:3, :3]
    K2_rect = rect_params['P2'][:3, :3]
    Q = rect_params['Q']
    baseline = -1.0 / Q[3, 2] if Q[3, 2] != 0 else float('inf')
    focal_length = (K1_rect[0, 0] + K1_rect[1, 1]) / 2
    
    params = {
        'source_model': {'type': 'double_sphere'},
        'pipeline': 'virtual_pinhole → cv2.stereoRectify → remap',
        'virtual_pinhole': {
            'image_size': list(image_size),
            'fov_deg': fov_deg,
            'K_left': K_pin_left.tolist(),
            'K_right': K_pin_right.tolist(),
            'distortion_model': 'none',
            'distortion_coeffs': []
        },
        'rectified': {
            'K1_rect': K1_rect.tolist(),
            'K2_rect': K2_rect.tolist(),
            'R1': rect_params['R1'].tolist(),
            'R2': rect_params['R2'].tolist(),
            'P1': rect_params['P1'].tolist(),
            'P2': rect_params['P2'].tolist(),
            'Q': Q.tolist(),
            'validPixROI1': list(rect_params['validPixROI1']),
            'validPixROI2': list(rect_params['validPixROI2'])
        },
        'stereo_params': {
            'baseline_m': float(baseline),
            'baseline_mm': float(baseline * 1000),
            'focal_length': float(focal_length),
            'depth_factor': float(focal_length * baseline)
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)
    print(f"\n保存校正参数: {output_path}")


def _save_rectify_params_equidistant(
    output_path, model_left, model_right, rect_params, image_size,
    crop_roi=None
):
    """保存 Equidistant 管线的校正参数"""
    K1_rect = rect_params['P1'][:3, :3]
    K2_rect = rect_params['P2'][:3, :3]
    Q = rect_params['Q']
    baseline = -1.0 / Q[3, 2] if Q[3, 2] != 0 else float('inf')
    focal_length = (K1_rect[0, 0] + K1_rect[1, 1]) / 2
    
    params = {
        'source_model': {'type': 'equidistant'},
        'pipeline': 'cv2.fisheye.stereoRectify → cv2.fisheye.initUndistortRectifyMap',
        'source_cameras': {
            'K_left': model_left.K.tolist(),
            'D_left': model_left.D.tolist(),
            'K_right': model_right.K.tolist(),
            'D_right': model_right.D.tolist(),
        },
        'rectified': {
            'image_size': list(image_size),
            'K1_rect': K1_rect.tolist(),
            'K2_rect': K2_rect.tolist(),
            'R1': rect_params['R1'].tolist(),
            'R2': rect_params['R2'].tolist(),
            'P1': rect_params['P1'].tolist(),
            'P2': rect_params['P2'].tolist(),
            'Q': Q.tolist(),
        },
        'stereo_params': {
            'baseline_m': float(baseline),
            'baseline_mm': float(baseline * 1000),
            'focal_length': float(focal_length),
            'depth_factor': float(focal_length * baseline)
        }
    }
    
    if crop_roi is not None:
        params['crop_roi'] = {
            'x_min': int(crop_roi[0]),
            'y_min': int(crop_roi[1]),
            'x_max': int(crop_roi[2]),
            'y_max': int(crop_roi[3]),
        }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)
    print(f"\n保存校正参数: {output_path}")


def _copy_files(input_dir, output_session_dir, camchain_path, calib_dir):
    """复制其他文件到输出目录"""
    print(f"\n复制其他文件...")
    
    # 确保输出目录存在
    os.makedirs(output_session_dir, exist_ok=True)
    
    # 复制 meta.json
    shutil.copy(input_dir / 'meta.json', output_session_dir / 'meta.json')
    
    # 复制 calibrations 目录
    calib_output_dir = output_session_dir / 'calibrations'
    os.makedirs(calib_output_dir, exist_ok=True)
    shutil.copy(camchain_path, calib_output_dir / camchain_path.name)
    
    for calib_file in calib_dir.iterdir():
        if calib_file.name != camchain_path.name:
            shutil.copy(calib_file, calib_output_dir / calib_file.name)
    
    # 复制 timestamps 目录
    timestamps_dir = input_dir / 'timestamps'
    if timestamps_dir.exists():
        timestamps_output_dir = output_session_dir / 'timestamps'
        os.makedirs(timestamps_output_dir, exist_ok=True)
        for ts_file in timestamps_dir.iterdir():
            shutil.copy(ts_file, timestamps_output_dir / ts_file.name)
    
    # 复制其他视频
    videos_dir = input_dir / 'videos'
    videos_output_dir = output_session_dir / 'videos'
    os.makedirs(videos_output_dir, exist_ok=True)
    for stream in ['bleft', 'bright', 'left', 'right']:
        try:
            src = find_named_video(videos_dir, stream)
        except FileNotFoundError:
            continue
        shutil.copy(src, videos_output_dir / f'{stream}.mp4')


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='双目立体校正脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动检测模型
  python scripts/rectify_stereo_video.py --input data/session_xxx --output data/output

  # Double Sphere: 通过 --fov 控制虚拟针孔视场角
  python scripts/rectify_stereo_video.py --input ... --output ... --model double_sphere --fov 120

  # Equidistant: 通过 --balance 控制裁剪/保留
  python scripts/rectify_stereo_video.py --input ... --output ... --model equidistant --balance 0.5
"""
    )
    
    # 通用参数
    common = parser.add_argument_group('通用参数')
    common.add_argument('--input', type=str, required=True,
                        help='输入 session 目录')
    common.add_argument('--output', type=str, required=True,
                        help='输出目录')
    common.add_argument('--width', type=int, default=1600,
                        help='输出图像宽度 (默认: 1600)')
    common.add_argument('--height', type=int, default=900,
                        help='输出图像高度 (默认: 900)')
    common.add_argument('--model', type=str, default=None,
                        choices=['double_sphere', 'equidistant', 'fisheye'],
                        help='手动指定相机模型类型 (默认: 自动检测)')
    common.add_argument('--camchain', type=str, default=None,
                        help='camchain 文件名 (默认: 自动查找)')
    
    # Double Sphere 专用参数
    ds_group = parser.add_argument_group(
        'Double Sphere 参数',
        '仅在 model=double_sphere 时生效'
    )
    ds_group.add_argument('--fov', type=float, default=120.0,
                          help='虚拟针孔水平视场角 (默认: 120度)')
    
    # Equidistant (Fisheye) 专用参数
    eq_group = parser.add_argument_group(
        'Equidistant (Fisheye) 参数',
        '仅在 model=equidistant 时生效'
    )
    eq_group.add_argument('--balance', type=float, default=0.0,
                          help='裁剪平衡参数 (0=裁剪到全部有效, 1=保留全部像素, 默认: 0)')
    
    # 共享立体校正参数（两个模型都有，但语义略有不同）
    rect_group = parser.add_argument_group(
        '立体校正参数',
        '两个模型共享，但语义不同'
    )
    rect_group.add_argument('--alpha', type=float, default=0.0,
                            help='Double Sphere: cv2.stereoRectify 的 alpha (0=裁剪, 1=保留)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    session_name = input_dir.name
    output_session_dir = output_dir / session_name
    
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_session_dir}")
    
    # ================= 1. 加载标定文件 =================
    calib_dir = input_dir / 'calibrations'
    
    if args.camchain:
        camchain_path = calib_dir / args.camchain
    else:
        camchain_candidates = [
            'camera01-camchain.yaml',
            'camera-camchain.yaml',
        ]
        camchain_path = None
        for candidate in camchain_candidates:
            path = calib_dir / candidate
            if path.exists():
                camchain_path = path
                break
        
        if camchain_path is None:
            raise FileNotFoundError(
                f"在 {calib_dir} 中找不到 camchain 文件。"
                f"尝试过: {camchain_candidates}"
            )
    
    print(f"\n加载标定文件: {camchain_path}")
    camchain = load_camchain(str(camchain_path))
    validate_stereo_camchain(camchain, args.model)
    
    # 获取模型类型
    model_type = args.model
    cam0_info = camchain['cam0']
    cam1_info = camchain['cam1']
    
    model_name_left = get_model_type_name(cam0_info, model_type)
    model_name_right = get_model_type_name(cam1_info, model_type)
    
    print(f"\n相机模型:")
    print(f"  左相机 (cam0): {model_name_left}")
    print(f"  右相机 (cam1): {model_name_right}")
    
    # 创建相机模型
    camera_model_left = create_camera_model(cam0_info, model_type)
    camera_model_right = create_camera_model(cam1_info, model_type)
    
    # 打印相机参数
    if isinstance(camera_model_left, DoubleSphereCameraModel):
        print(f"  左相机参数: fx={camera_model_left.fx:.2f}, fy={camera_model_left.fy:.2f}, "
              f"xi={camera_model_left.xi:.4f}, alpha={camera_model_left.alpha:.4f}")
    elif isinstance(camera_model_left, EquidistantCameraModel):
        print(f"  左相机参数: fx={camera_model_left.fx:.2f}, fy={camera_model_left.fy:.2f}, "
              f"k1={camera_model_left.k1:.4f}, k2={camera_model_left.k2:.4f}, "
              f"k3={camera_model_left.k3:.4f}, k4={camera_model_left.k4:.4f}")
    
    if isinstance(camera_model_right, DoubleSphereCameraModel):
        print(f"  右相机参数: fx={camera_model_right.fx:.2f}, fy={camera_model_right.fy:.2f}, "
              f"xi={camera_model_right.xi:.4f}, alpha={camera_model_right.alpha:.4f}")
    elif isinstance(camera_model_right, EquidistantCameraModel):
        print(f"  右相机参数: fx={camera_model_right.fx:.2f}, fy={camera_model_right.fy:.2f}, "
              f"k1={camera_model_right.k1:.4f}, k2={camera_model_right.k2:.4f}, "
              f"k3={camera_model_right.k3:.4f}, k4={camera_model_right.k4:.4f}")
    
    # 解析外参 (cam1 -> cam0)
    R, t = parse_extrinsic(cam1_info)
    print(f"\n外参 (cam1->cam0):")
    print(f"  R:\n{R}")
    print(f"  t: {t}")
    print(f"  基线长度: {np.linalg.norm(t)*1000:.2f} mm")
    
    # 复制文件
    _copy_files(input_dir, output_session_dir, camchain_path, calib_dir)
    
    # 读取 meta.json 获取 fps
    meta_path = input_dir / 'meta.json'
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    fps = meta.get('fps', 30)
    print(f"\n视频 FPS: {fps}")
    
    output_width, output_height = args.width, args.height
    
    # ================= 2. 选择管线并运行 =================
    if isinstance(camera_model_left, EquidistantCameraModel):
        # Equidistant 管线 (使用 --balance 参数, 始终用原始分辨率)
        rect_params, actual_size = run_equidistant_pipeline(
            camera_model_left, camera_model_right,
            R, t,
            output_width, output_height, args.balance,
            input_dir, output_session_dir, fps
        )
        actual_model_type = 'equidistant'
    else:
        # Double Sphere 管线
        rect_params = run_double_sphere_pipeline(
            camera_model_left, camera_model_right,
            R, t,
            output_width, output_height, args.fov, args.alpha,
            input_dir, output_session_dir, fps
        )
        actual_model_type = 'double_sphere'
        actual_size = (output_width, output_height)
    
    # ================= 3. 更新 meta.json =================
    print(f"\n更新 meta.json...")
    meta['rectified'] = True
    meta['rectified_resolution'] = list(actual_size)
    meta['source_model'] = actual_model_type
    if actual_model_type == 'double_sphere':
        meta['rectified_fov'] = args.fov
    with open(output_session_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    # ================= 4. 输出总结 =================
    Q = rect_params['Q']
    baseline = -1.0 / Q[3, 2]
    K1_rect = rect_params['P1'][:3, :3]
    focal_length = (K1_rect[0, 0] + K1_rect[1, 1]) / 2
    depth_factor = focal_length * baseline
    
    print(f"\n{'='*60}")
    print(f"立体校正完成！")
    print(f"{'='*60}")
    print(f"使用的相机模型: {actual_model_type}")
    if actual_model_type == 'equidistant':
        print(f"管线: cv2.fisheye.stereoRectify → cv2.fisheye.initUndistortRectifyMap")
    else:
        print(f"管线: 虚拟针孔 → cv2.stereoRectify → remap")
    print(f"输出目录: {output_session_dir}")
    print(f"左相机校正视频: {output_session_dir / 'videos' / 'left_rectified.mp4'}")
    print(f"右相机校正视频: {output_session_dir / 'videos' / 'right_rectified.mp4'}")
    print(f"校正参数: {output_session_dir / 'calibrations' / 'rectify_params.yaml'}")
    
    print(f"\n深度估计参数:")
    print(f"  基线: {baseline*1000:.2f} mm")
    print(f"  焦距: {focal_length:.2f}")
    print(f"  depth = {depth_factor:.4f} / disparity")


if __name__ == '__main__':
    main()
