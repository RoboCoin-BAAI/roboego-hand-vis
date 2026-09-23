import numpy as np
import cv2

class DoubleSphereCameraModel():
    def __init__(self, fx, fy, cx, cy, xi, alpha):
        self.fx = fx; self.fy = fy; self.cx = cx; self.cy = cy
        self.xi = xi; self.alpha = alpha

    def double_sphere_project(self, points_3d):
        """双球模型投影（批量），无效区域返回 -1 触发 BORDER_CONSTANT"""
        x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
        
        d1 = np.sqrt(x**2 + y**2 + z**2)
        k = self.xi * d1 + z
        d2 = np.sqrt(x**2 + y**2 + k**2)
        denom = self.alpha * d2 + (1 - self.alpha) * k
        
        # 1e-6 防止数值不稳定导致的除零或极大坐标
        valid = denom > 1e-6
        u = np.full_like(denom, -1.0, dtype=np.float64)
        v = np.full_like(denom, -1.0, dtype=np.float64)
        
        mx = x[valid] / denom[valid]
        my = y[valid] / denom[valid]
        u[valid] = self.fx * mx + self.cx
        v[valid] = self.fy * my + self.cy
        
        return np.column_stack((u, v))
    
    def generate_pinhole_to_double_sphere_map(
        self, fx_pin, fy_pin, cx_pin, cy_pin, output_width, output_height
    ):
        u_grid, v_grid = np.meshgrid(np.arange(output_width), np.arange(output_height))
        u_flat = u_grid.flatten().astype(np.float64)
        v_flat = v_grid.flatten().astype(np.float64)
        
        mx = (u_flat - cx_pin) / fx_pin
        my = (v_flat - cy_pin) / fy_pin
        mz = np.ones_like(mx)
        norm = np.sqrt(mx*mx + my*my + mz*mz)
        rays = np.column_stack((mx/norm, my/norm, mz/norm))
        
        pixels_ds = self.double_sphere_project(rays)
        
        map_x = pixels_ds[:, 0].reshape(output_height, output_width).astype(np.float32)
        map_y = pixels_ds[:, 1].reshape(output_height, output_width).astype(np.float32)
        return map_x, map_y
    
    def remap_frame(self, image_ds, map_x, map_y):
        return cv2.remap(image_ds, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    def undistort(self, image_ds, fx_pin, fy_pin, cx_pin, cy_pin, output_width, output_height):
        map_x, map_y = self.generate_pinhole_to_double_sphere_map(fx_pin, fy_pin, cx_pin, cy_pin, output_width, output_height)
        return self.remap_frame(image_ds, map_x, map_y)
