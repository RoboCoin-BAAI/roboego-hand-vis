import numpy as np
import cv2

class DoubleSphereCameraModel():
    def __init__(self, fx, fy, cx, cy, xi, alpha):
        values = np.asarray([fx, fy, cx, cy, xi, alpha], dtype=float)
        if not np.isfinite(values).all() or min(fx, fy) <= 0 or not 0 <= alpha <= 1 or not -1 < xi <= 1:
            raise ValueError('Invalid DS parameters: positive focal lengths, -1 < xi <= 1, 0 <= alpha <= 1 required')
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
        # DS injective projection domain, in addition to a nonzero denominator.
        w1 = self.alpha / (1-self.alpha) if self.alpha <= .5 else (1-self.alpha) / self.alpha
        w2 = (w1+self.xi) / np.sqrt(2*w1*self.xi+self.xi*self.xi+1)
        valid = (denom > 1e-6) & (z > -w2*d1) & np.isfinite(points_3d).all(axis=1)
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

    @staticmethod
    def stereo_rectify(model_left, model_right, image_size, R, t,
                       new_image_size=None, balance=0.0):
        """Rectify virtual pinhole views; DS projection is used only in the remap.

        A conservative central-ray focal length is used at balance=0. Increasing
        balance widens the virtual view up to 2x; the shared valid-ROI stage crops
        unavailable source pixels. This is not OpenCV fisheye's balance policy.
        """
        if not np.isfinite(balance) or not 0 <= balance <= 1:
            raise ValueError('balance must lie in [0, 1]')
        w, h = image_size
        focal = min(m.fx/(1+m.xi) for m in (model_left, model_right)) / (1+balance)
        K = np.array([[focal, 0., (w-1)/2], [0., focal, (h-1)/2], [0., 0., 1.]])
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K, np.zeros(5), K, np.zeros(5), image_size,
            np.ascontiguousarray(R, dtype=np.float64), np.array(t, dtype=np.float64).reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
            newImageSize=new_image_size or image_size)
        return dict(R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)

    def generate_stereo_remap(self, R_rect, P_rect, output_width, output_height):
        """Map each rectified pinhole pixel back into the original DS camera."""
        u, v = np.meshgrid(np.arange(output_width), np.arange(output_height))
        pixels = np.column_stack((u.ravel(), v.ravel(), np.ones(u.size)))
        K_rect = np.asarray(P_rect, dtype=float)[:3, :3]
        # Column-vector convention: ray_raw = R_rect.T @ inv(K_rect) @ pixel.
        rays = pixels @ np.linalg.inv(K_rect).T @ np.asarray(R_rect, dtype=float)
        uv = self.double_sphere_project(rays)
        return tuple(uv[:, i].reshape(output_height, output_width).astype(np.float32) for i in (0, 1))
    
    def undistort(self, image_ds, fx_pin, fy_pin, cx_pin, cy_pin, output_width, output_height):
        map_x, map_y = self.generate_pinhole_to_double_sphere_map(fx_pin, fy_pin, cx_pin, cy_pin, output_width, output_height)
        return self.remap_frame(image_ds, map_x, map_y)
