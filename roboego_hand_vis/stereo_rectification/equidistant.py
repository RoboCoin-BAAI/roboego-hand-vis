import numpy as np
import cv2


class EquidistantCameraModel:
    """
    等距投影（鱼眼）相机模型 - OpenCV fisheye 模型
    
    使用 Kannala-Brandt 投影模型：
    r = f * theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
    其中 theta 是入射角，r 是图像上的半径
    """
    
    def __init__(self, fx, fy, cx, cy, k1, k2, k3, k4):
        """
        初始化等距投影相机模型
        
        Args:
            fx, fy: 焦距
            cx, cy: 主点坐标
            k1, k2, k3, k4: 畸变系数
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        
        # OpenCV 内参矩阵
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # OpenCV 畸变系数向量
        self.D = np.array([[k1, k2, k3, k4]], dtype=np.float64)
    
    def generate_remap_with_cv2(
        self, fx_pin, fy_pin, cx_pin, cy_pin, output_width, output_height,
        R=None, P=None
    ):
        """
        使用 OpenCV 内置函数生成去畸变映射
        
        这个方法使用 cv2.fisheye.initUndistortRectifyMap，
        可以同时处理去畸变和立体校正。
        
        Args:
            fx_pin, fy_pin: 虚拟针孔相机焦距
            cx_pin, cy_pin: 虚拟针孔相机主点
            output_width, output_height: 输出图像尺寸
            R: 旋转校正矩阵（用于立体校正）
            P: 投影矩阵（用于立体校正）
            
        Returns:
            map_x, map_y: remap 映射表
        """
        # 构建新的内参矩阵
        K_new = np.array([
            [fx_pin, 0, cx_pin],
            [0, fy_pin, cy_pin],
            [0, 0, 1]
        ], dtype=np.float64)
        
        if R is None:
            R = np.eye(3, dtype=np.float64)
        if P is None:
            P = K_new
        
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, R, P, (output_width, output_height), cv2.CV_32FC1
        )
        
        return map_x, map_y
    
    def remap_frame(self, image, map_x, map_y):
        """
        应用 remap 映射到图像
        
        Args:
            image: 输入图像
            map_x, map_y: 映射表
            
        Returns:
            重映射后的图像
        """
        return cv2.remap(
            image, map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
    
    def undistort(self, image, fx_pin, fy_pin, cx_pin, cy_pin, output_width, output_height):
        """
        去畸变：将鱼眼图像转换为虚拟针孔图像
        
        Args:
            image: 输入鱼眼图像
            fx_pin, fy_pin: 虚拟针孔相机焦距
            cx_pin, cy_pin: 虚拟针孔相机主点
            output_width, output_height: 输出图像尺寸
            
        Returns:
            去畸变后的图像
        """
        map_x, map_y = self.generate_remap_with_cv2(
            fx_pin, fy_pin, cx_pin, cy_pin, output_width, output_height
        )
        return self.remap_frame(image, map_x, map_y)

    @staticmethod
    def stereo_rectify(model_left, model_right, image_size, R, t,
                       new_image_size=None, balance=0.0):
        """
        使用 OpenCV fisheye 模块进行立体校正
        
        直接调用 cv2.fisheye.stereoRectify()，使用实际的鱼眼相机参数，
        不需要虚拟针孔。
        
        Args:
            model_left: 左相机 EquidistantCameraModel
            model_right: 右相机 EquidistantCameraModel
            image_size: 原始图像尺寸 (width, height)
            R: 右相机相对于左相机的旋转矩阵 (3x3)
            t: 右相机相对于左相机的平移向量 (3x1)
            new_image_size: 校正后图像尺寸，None 表示使用原始尺寸
            balance: 设置 newImageSize (0=裁剪到全部有效, 1=保留全部像素)
            
        Returns:
            dict: 包含 R1, R2, P1, P2, Q
        """
        if new_image_size is None:
            new_image_size = image_size
        
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
            model_left.K, model_left.D,
            model_right.K, model_right.D,
            image_size, R, t,
            flags=cv2.CALIB_ZERO_DISPARITY,
            newImageSize=new_image_size,
            balance=balance
        )
        
        return {
            'R1': R1, 'R2': R2,
            'P1': P1, 'P2': P2,
            'Q': Q,
        }

    def generate_stereo_remap(
        self, R_rect, P_rect, output_width, output_height
    ):
        """
        生成立体校正后的 remap map
        
        使用 cv2.fisheye.initUndistortRectifyMap()，
        同时完成去畸变和立体校正。
        
        Args:
            R_rect: 立体校正旋转矩阵 (来自 stereo_rectify 的 R1 或 R2)
            P_rect: 立体校正投影矩阵 (来自 stereo_rectify 的 P1 或 P2)
            output_width, output_height: 输出图像尺寸
            
        Returns:
            map_x, map_y: remap 映射表
        """
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, R_rect, P_rect,
            (output_width, output_height), cv2.CV_32FC1
        )
        return map_x, map_y
