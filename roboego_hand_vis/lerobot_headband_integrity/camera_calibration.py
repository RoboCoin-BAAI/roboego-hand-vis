"""Model-aware structural checks; passing does not certify calibration accuracy."""
import math


def camera_error(camera):
    if not isinstance(camera, dict):
        return 'Camera calibration must be a mapping'

    def numbers(values, count):
        return (isinstance(values, list) and len(values) == count and
                all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in values))

    resolution = camera.get('resolution')
    if not numbers(resolution, 2) or any(v <= 0 or int(v) != v for v in resolution):
        return 'resolution must contain two positive integer dimensions'
    model = camera.get('camera_model', 'pinhole')
    distortion_model = camera.get('distortion_model', 'equidistant')
    intrinsics = camera.get('intrinsics')
    distortion = camera.get('distortion_coeffs')
    if model == 'ds':
        if not numbers(intrinsics, 6):
            return 'ds requires six finite intrinsics: xi, alpha, fx, fy, cx, cy'
        if not 0 <= intrinsics[1] <= 1:
            return 'ds alpha must lie in [0, 1]'
        if min(intrinsics[2:4]) <= 0:
            return 'ds fx and fy must be positive (xi may be negative)'
        if distortion_model != 'none' or not numbers(distortion, 0):
            return 'ds requires distortion_model=none and distortion_coeffs=[]'
    elif model == 'pinhole':
        if not numbers(intrinsics, 4) or min(intrinsics[:2]) <= 0:
            return 'pinhole requires finite fx, fy, cx, cy with positive focal lengths'
        counts = {'equidistant': 4, 'radtan': 4, 'none': 0}
        if distortion_model not in counts:
            return f'Unsupported pinhole distortion model: {distortion_model}'
        if not numbers(distortion, counts[distortion_model]):
            return f'pinhole/{distortion_model} requires {counts[distortion_model]} finite distortion coefficients'
    else:
        return f'Unsupported camera model: {model}'
    return None
