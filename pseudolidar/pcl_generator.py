from pathlib import Path
from typing import Tuple, Union, Optional

import numpy as np
from numpy.matlib import repmat
from skimage import io

FAR = 1000.0  # max depth in meters


class PointCloudGenerator(object):
    """
    Generator for creating local point clouds from depth maps
    """

    def __init__(self,
                 fov: Optional[int] = None,
                 image_dims: Optional[Tuple[int, int]] = None,
                 intrinsics: Optional[np.array] = None) -> None:
        """
        Initialize the point cloud generator.
        The intrinsics can either be set by the field of view + the image dimensions (CARLA),
        or by directly providing an intrinsics matrix.

        :param fov: Field of View of the camera
        :param image_dims: A tuple with the size of the image
        :param intrinsics: A 3x3 numpy array representing the camera intrinsics
        """
        if intrinsics is not None:
            self.intrinsics = intrinsics

        elif fov is not None and image_dims is not None:
            self.fov = fov
            self.image_dims = image_dims
            self.intrinsics = self.get_carla_camera_intrinsics(
                image_dims, fov)

        else:
            raise ValueError(
                'Either intrinsics matrix or FOV + image dimensions must be given!')

    def generate(self, depth: Union[str, np.array], max_depth: int = 1.0, sparsity: int = 1):
        """
        Generate a local point cloud from a depth map.

        :param depth: The depth map as NxM numpy array or path to CARLA encoded file
        :param max_depth: The maximum depth of pixels. Pixels with higher depths are dropped.
        :param sparsity: Stride used for back-projection. Must be an integer >= 1
        :return pcl: The resulting point cloud as Nx3 numpy array
        """
        if isinstance(depth, str) or isinstance(depth, Path):
            depth = self.read_depth(depth)

        # make depth map sparse to imitate lidar signal
        tmp = depth[::sparsity, ::sparsity]
        depth = np.ones_like(depth)
        depth[::sparsity, ::sparsity] = tmp

        pcl = self.depth_to_local_point_cloud(depth, max_depth=max_depth)
        return pcl

    def depth_to_local_point_cloud(self, depth_map: np.array, max_depth: float) -> np.array:
        """
        Convert a depth map into a point cloud

        :param depth_map: The depth map as NxM numpy array
        :param max_depth: The maximum depth of pixels. Pixels with higher depths are dropped.
        :return pcl: The resulting point cloud as Nx3 numpy array
        """

        normalized_depth = depth_map  # depth_to_array(image)
        height, width = depth_map.shape

        # 2d pixel coordinates
        pixel_length = width * height
        u_coord = repmat(np.r_[width-1:-1:-1],
                         height, 1).reshape(pixel_length)
        v_coord = repmat(np.c_[height-1:-1:-1],
                         1, width).reshape(pixel_length)

        normalized_depth = np.reshape(normalized_depth, pixel_length)

        # Search for pixels where the depth is greater than max_depth to
        # delete them
        max_depth_indexes = np.where(normalized_depth > max_depth)
        normalized_depth = np.delete(normalized_depth, max_depth_indexes)
        u_coord = np.delete(u_coord, max_depth_indexes)
        v_coord = np.delete(v_coord, max_depth_indexes)

        # pd2 = [u,v,1]
        p2d = np.array([u_coord, v_coord, np.ones_like(u_coord)])
        
        # P = [X,Y,Z]
        p3d = np.dot(np.linalg.inv(self.intrinsics), p2d)
        p3d *= normalized_depth * FAR
        p3d = np.transpose(p3d, (1, 0))
        p3d = p3d[:, [2, 0, 1]]
        
        return p3d

    @staticmethod
    def get_carla_camera_intrinsics(img_dims: Tuple[int, int], fov: int) -> np.array:
        """
        Compute camera intrinsics for CARLA images.

        :param img_dims: A tuple with the dimensions of the image
        :param fov: The field of view
        :return K: the camera intrinsics as 3x3 numpy array
        """
        f = img_dims[1] / (2. * np.tan(fov * np.pi / 360.))
        center_x = img_dims[1] // 2
        center_y = img_dims[0] // 2
        K = np.array([[f, 0, center_x],
                      [0, f, center_y],
                      [0, 0, 1]])

        return K

    @staticmethod
    def read_depth(depth_file: str) -> np.array:
        """
        Reference: https://carla.readthedocs.io/en/stable/cameras_and_sensors/#camera-depth-map

        :param depth_file: The path to the CARLA encoded depth map
        :return depth: The depth map as normalized NxM numpy array
        """
        depth = io.imread(depth_file)

        return PointCloudGenerator.process_depth(depth)
    
    @staticmethod
    def process_depth(depth: np.array) -> np.array:
        """
        Decode a 3-channel depth map into normalized values

        :param depth: NxMx3 encoded depth map as numpy array
        :return depth: NxM normalized depth map as numpy array
        """
        depth = depth[:, :, 0] * 1.0 + depth[:, :, 1] * \
            256.0 + depth[:, :, 2] * (256.0 * 256)
        depth = depth * (1 / (256 * 256 * 256 - 1))
        return depth
