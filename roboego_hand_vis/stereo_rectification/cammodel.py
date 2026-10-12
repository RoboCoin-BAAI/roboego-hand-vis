"""
相机模型工厂模块

提供根据标定文件自动创建相机模型的功能。
"""

import yaml
from typing import Union

try:
    from .double_sphere import DoubleSphereCameraModel
    from .equidistant import EquidistantCameraModel
except ImportError:
    from double_sphere import DoubleSphereCameraModel
    from equidistant import EquidistantCameraModel


def load_camchain(camchain_path: str) -> dict:
    """
    加载 camchain.yaml 标定文件
    
    Args:
        camchain_path: camchain 文件路径
        
    Returns:
        标定数据字典
    """
    with open(camchain_path, 'r') as f:
        camchain = yaml.safe_load(f)
    return camchain


def validate_camera_model(cam_info: dict, model_type: str = None) -> str:
    """Reject unsupported calibration before constructing a remap model."""
    if not isinstance(cam_info, dict):
        raise ValueError("相机标定必须是字典")
    camera = str(cam_info.get('camera_model', '')).lower().strip()
    distortion = str(cam_info.get('distortion_model', '')).lower().strip()
    requested = str(model_type).lower().strip() if model_type is not None else None
    ds_names = {'ds', 'double_sphere', 'doublesphere', 'double-sphere'}
    if camera in ds_names or distortion in ds_names or requested in ds_names:
        raise ValueError("双球面模型的去畸变/立体矫正实现尚未完成，当前不支持运行")
    if camera != 'pinhole' or distortion != 'equidistant':
        raise ValueError(
            f"当前仅支持 camera_model=pinhole + distortion_model=equidistant，"
            f"实际为 camera_model={camera!r}, distortion_model={distortion!r}"
        )
    if requested not in (None, 'equidistant', 'fisheye', 'kannala_brandt'):
        raise ValueError(f"指定模型 {requested!r} 与标定 pinhole + equidistant 不匹配")
    intrinsics = cam_info.get('intrinsics')
    coeffs = cam_info.get('distortion_coeffs')
    if not isinstance(intrinsics, (list, tuple)) or len(intrinsics) != 4:
        raise ValueError("pinhole + equidistant 需要 4 个内参 [fx, fy, cx, cy]")
    if not isinstance(coeffs, (list, tuple)) or len(coeffs) != 4:
        raise ValueError("equidistant 需要 4 个畸变系数 [k1, k2, k3, k4]")
    return 'equidistant'


def validate_stereo_camchain(camchain: dict, model_type: str = None) -> str:
    """Check both stereo cameras; unsupported models must not fall back."""
    if not isinstance(camchain, dict):
        raise ValueError("camchain 标定必须是字典")
    for name in ('cam0', 'cam1'):
        try:
            validate_camera_model(camchain.get(name), model_type)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc
    return 'equidistant'


def create_camera_model(
    cam_info: dict, 
    model_type: str = None
) -> Union[DoubleSphereCameraModel, EquidistantCameraModel]:
    """
    根据 camchain 配置自动创建相机模型

    当前入口先严格校验 camera_model=pinhole 与 distortion_model=equidistant。
    双球面实现尚未完成，下面保留的双球面/旧推断分支暂不开放。
    
    支持的模型类型:
    - 'double_sphere': 双球面模型 (Double Sphere)
    - 'equidistant': 等距投影模型 (OpenCV fisheye / Kannala-Brandt)
    
    如果 model_type 未指定，则根据 cam_info 中的字段自动检测:
    - 如果存在 'distortion_model' 字段且值为 'equidistant'，使用等距投影模型
    - 如果存在 'distortion_model' 字段且值为 'double_sphere'，使用双球面模型
    - 如果内参有 6 个参数，假设为双球面模型
    - 如果内参有 4 个参数且有畸变系数，假设为等距投影模型
    
    Args:
        cam_info: camchain 中单个相机的配置字典，包含:
            - intrinsics: 内参列表
            - distortion_model: 畸变模型名称 (可选)
            - distortion_coeffs: 畸变系数 (可选)
        model_type: 手动指定模型类型，覆盖自动检测
        
    Returns:
        相机模型实例 (DoubleSphereCameraModel 或 EquidistantCameraModel)
        
    Raises:
        ValueError: 无法识别的模型类型
    """
    # The double-sphere implementation is unfinished. Never infer/override an
    # unsupported calibration from parameter counts alone.
    model_type = validate_camera_model(cam_info, model_type)
    intrinsics = cam_info.get('intrinsics', [])
    distortion_model = cam_info.get('distortion_model', '')
    distortion_coeffs = cam_info.get('distortion_coeffs', [])
    
    # 如果指定了模型类型，优先使用
    if model_type is not None:
        model_type = model_type.lower().strip()
        
        if model_type == 'double_sphere':
            if len(intrinsics) != 6:
                raise ValueError(
                    f"双球面模型需要 6 个内参 [xi, alpha, fx, fy, cx, cy]，"
                    f"但得到 {len(intrinsics)} 个"
                )
            xi, alpha, fx, fy, cx, cy = intrinsics
            return DoubleSphereCameraModel(fx, fy, cx, cy, xi, alpha)
            
        elif model_type in ('equidistant', 'fisheye', 'kannala_brandt'):
            if len(intrinsics) != 4:
                raise ValueError(
                    f"等距投影模型需要 4 个内参 [fx, fy, cx, cy]，"
                    f"但得到 {len(intrinsics)} 个"
                )
            if len(distortion_coeffs) != 4:
                raise ValueError(
                    f"等距投影模型需要 4 个畸变系数 [k1, k2, k3, k4]，"
                    f"但得到 {len(distortion_coeffs)} 个"
                )
            fx, fy, cx, cy = intrinsics
            k1, k2, k3, k4 = distortion_coeffs
            return EquidistantCameraModel(fx, fy, cx, cy, k1, k2, k3, k4)
            
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 自动检测模型类型
    distortion_model_lower = distortion_model.lower().strip() if distortion_model else ''
    
    # 根据 distortion_model 字段判断
    if distortion_model_lower == 'equidistant':
        if len(intrinsics) != 4:
            raise ValueError(
                f"等距投影模型需要 4 个内参 [fx, fy, cx, cy]，"
                f"但得到 {len(intrinsics)} 个"
            )
        if len(distortion_coeffs) != 4:
            raise ValueError(
                f"等距投影模型需要 4 个畸变系数 [k1, k2, k3, k4]，"
                f"但得到 {len(distortion_coeffs)} 个"
            )
        fx, fy, cx, cy = intrinsics
        k1, k2, k3, k4 = distortion_coeffs
        return EquidistantCameraModel(fx, fy, cx, cy, k1, k2, k3, k4)
        
    elif distortion_model_lower in ('double_sphere', 'ds', 'doublesphere'):
        if len(intrinsics) != 6:
            raise ValueError(
                f"双球面模型需要 6 个内参 [xi, alpha, fx, fy, cx, cy]，"
                f"但得到 {len(intrinsics)} 个"
            )
        xi, alpha, fx, fy, cx, cy = intrinsics
        return DoubleSphereCameraModel(fx, fy, cx, cy, xi, alpha)
    
    # 根据内参数量推断
    if len(intrinsics) == 6:
        # 6 个参数: xi, alpha, fx, fy, cx, cy (双球面)
        xi, alpha, fx, fy, cx, cy = intrinsics
        return DoubleSphereCameraModel(fx, fy, cx, cy, xi, alpha)
        
    elif len(intrinsics) == 4 and len(distortion_coeffs) == 4:
        # 4 个内参 + 4 个畸变系数: 等距投影
        fx, fy, cx, cy = intrinsics
        k1, k2, k3, k4 = distortion_coeffs
        return EquidistantCameraModel(fx, fy, cx, cy, k1, k2, k3, k4)
        
    else:
        raise ValueError(
            f"无法自动检测相机模型类型。"
            f"内参数量: {len(intrinsics)}，畸变系数数量: {len(distortion_coeffs)}。"
            f"请指定 model_type 参数或检查标定文件格式。"
        )


def get_model_type_name(cam_info: dict, model_type: str = None) -> str:
    """
    获取相机模型类型名称（用于日志输出）
    
    Args:
        cam_info: camchain 中单个相机的配置字典
        model_type: 手动指定的模型类型
        
    Returns:
        模型类型名称字符串
    """
    if model_type is not None:
        return model_type
    
    distortion_model = cam_info.get('distortion_model', '').lower().strip()
    intrinsics = cam_info.get('intrinsics', [])
    
    if distortion_model == 'equidistant':
        return 'equidistant (fisheye)'
    elif distortion_model in ('double_sphere', 'ds', 'doublesphere'):
        return 'double_sphere'
    elif len(intrinsics) == 6:
        return 'double_sphere (auto-detected)'
    elif len(intrinsics) == 4:
        return 'equidistant (auto-detected)'
    else:
        return 'unknown'
