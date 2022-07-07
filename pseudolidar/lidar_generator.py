import re
from tkinter.messagebox import NO
from typing import List


import numpy as np
import trimesh
import open3d as o3d


class PseudoLidarGenerator(object):
    """
    Generator for fusing multiple local point clouds into a single point cloud given the corresponding camera poses.
    """

    def __init__(self, extrinsics: List[np.array]) -> None:
        """
        :param extrinsics: A list of 4x4 numpy arrays containing the pose of the cameras
        """
        self.extrinsics = extrinsics

    def generate(self, local_pcls: List[np.array]) -> np.array:
        """
        Combine multiple local pointclouds into a single point cloud in a reference frame

        :param local_pcls: A list of Nx3 numpy arrays representing the local point clouds
        :return world_pcl: A Nx3 numpy array, containing all unique points projected into the reference frame
        """

        # assert argument is of the right length
        if len(local_pcls) != len(self.extrinsics):
            raise ValueError(
                'Lenght of image list must match length of extrinsics')

        # project each point cloud into the reference frame
        pcl_parts = []
        for idx, local_pcl in enumerate(local_pcls):
            pcl = self.project_pointcloud_to_world_frame(
                local_pcl, self.extrinsics[idx])

            pcl_parts.append(pcl)

        # concatenate point clouds and remove duplicates
        world_pcl = np.concatenate(tuple(pcl_parts))
        world_pcl = np.unique(world_pcl, axis=0)

        return world_pcl

    @staticmethod
    def post_process(pcl: np.array) -> np.array:
        """
        Drop features above a certain height.
        """
        pcl = pcl[(pcl[:, 2] <= 3.) & (pcl[:, 0] >= 0.)]

        return PseudoLidarGenerator.project_to_bev(pcl)

    @staticmethod
    def project_to_bev(pcl: np.array) -> np.array:
        """
        Projcet a point cloud to bird eye view.

        :param pcl: The point cloud as Nx3 numpy array
        :return img: 2x256x256 pseudo image representing the histogram features
        """
        _pcl = pcl[:, [1, 0, 2]]
        _pcl[:, 1] *= -1.

        hist = PseudoLidarGenerator.lidar_to_histogram_features(_pcl)
        #hist = np.transpose(hist, (0, 2, 1))

        return hist

    @staticmethod
    def to_grayscale_image(bev: np.array) -> np.array:
        """
        Project the BEV histogram to a grayscale image for visualization.

        :param bev: The 2xNxM BEV histogram
        :return img: The grayscale image as NxM numpy array
        """
        img = np.max(bev, axis=0)
        return img

    @staticmethod
    def project_pointcloud_to_world_frame(pcl: np.array, pose: np.array) -> np.array:
        """
        Project a point cloud to the world frame using its inverse pose.

        :param pcl: The point cloud as Nx3 numpy array
        :param pose: The pose as 4x4 numpy array
        :return pcl_world: The pointcloud in the world frame as Nx3 numpy array
        """

        T_inv = np.linalg.inv(pose)
        pcl = trimesh.points.PointCloud(pcl)
        pcl.apply_transform(T_inv)
        return pcl.vertices

    @staticmethod
    def lidar_to_histogram_features(pcl: np.array) -> np.array:
        """
        Convert LiDAR point cloud into 2-bin histogram over 256x256 grid

        Reference: https://github.com/autonomousvision/transfuser/blob/main/transfuser/data.py

        :param pcl: The input point cloud as Nx3 numpy array
        :return features: The histogram as 2x256x256 numpy array
        """
        def splat_points(point_cloud):
            # 256 x 256 grid
            pixels_per_meter = 16
            hist_max_per_pixel = 5
            x_meters_max = 8
            y_meters_max = 16
            xbins = np.linspace(-2*x_meters_max, 2*x_meters_max+1,
                                2*x_meters_max*pixels_per_meter+1)
            ybins = np.linspace(-y_meters_max, 0,
                                y_meters_max*pixels_per_meter+1)
            hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
            hist[hist > hist_max_per_pixel] = hist_max_per_pixel
            overhead_splat = hist/hist_max_per_pixel
            return overhead_splat

        below = pcl[pcl[..., 2] <= -2.0]
        above = pcl[pcl[..., 2] > -2.0]
        below_features = splat_points(below)
        above_features = splat_points(above)
        features = np.stack([below_features, above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)

        return features
    
    @staticmethod
    def reconstruct_mesh(pc: np.array, voxel_size: float=0.5) -> o3d.geometry.TriangleMesh:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
        pc.estimate_normals()
        pc = pc.voxel_down_sample(voxel_size=0.5)

        radii = [0.5]
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Warning) as cm:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pc, o3d.utility.DoubleVector(radii))

        return mesh
    
    @staticmethod
    def raycast_mesh(mesh: o3d.geometry.TriangleMesh, channels: int = 32, n_measurements: int = 250, origin: np.array = None):
        num_rays = channels * n_measurements
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

        rays = np.empty((n_measurements, channels, 6))
        horizontal_angles = np.linspace(-np.pi / 2, np.pi / 2, num=rays.shape[0])
        vertical_angles = np.linspace(-np.pi / 32, -np.pi / 2.5, num=rays.shape[1])

        ray_horizonatal_directions = np.array([np.cos(horizontal_angles), np.sin(horizontal_angles), np.zeros_like(horizontal_angles)]).T
        ray_vertical_directions = np.array([np.zeros_like(vertical_angles), np.zeros_like(vertical_angles), np.sin(vertical_angles)]).T

        # set ray origin (equal for all rays)
        if origin is not None:
            rays[:, :, :3] = origin
        else:
            rays[:, :, :3] = np.array([0., 0., 1.])

        for i in range(len(ray_horizonatal_directions)):
            for j in range(len(ray_vertical_directions)):
                direction = ray_horizonatal_directions[i] + ray_vertical_directions[j]
                direction /= np.linalg.norm(ray_horizonatal_directions[i] + ray_vertical_directions[j])
                rays[i, j, 3:] = direction

        rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        ans_dict = scene.cast_rays(rays)

        rays = rays.numpy()
        depth = ans_dict['t_hit'].numpy().reshape(-1)
        rays = rays.reshape((num_rays, 6))

        points = rays[:, :3] + (rays[:, 3:].T * depth).T
        points = np.delete(points, np.isinf(depth), axis=0)

        return points

    @staticmethod
    def load_pose(extrinsics):
        sinval = np.sin(extrinsics['yaw'] * np.pi / 180.)
        cosval = np.cos(extrinsics['yaw'] * np.pi / 180.)
        Rz = np.array([[cosval, -sinval, 0],
                    [sinval, cosval, 0],
                    [0, 0, 1]])
        t = np.array([extrinsics['x'], extrinsics['y'], extrinsics['z']])

        T = np.eye(4)
        T[:3, :3] = Rz
        T[:3, 3] = Rz @ t
        return T
