"""
registration_algs.py

Author: James Daniels

"""
import open3d as o3d
import numpy as np
from typing import Tuple, List
from utils import make_open3d_point_cloud

def downsample_point_cloud(pcd: o3d.geometry.PointCloud, 
                           voxel_size: float) -> o3d.geometry.PointCloud:
    """
    Downsample a point cloud using a voxel grid filter.

    Parameters:
    - pcd: The original point cloud.
    - voxel_size: The size of the voxel grid.

    Returns:
    - The downsampled point cloud.
    """
    return pcd.voxel_down_sample(voxel_size)


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, 
                           voxel_size: float) -> Tuple[o3d.geometry.PointCloud, 
                                                       o3d.pipelines.registration.Feature]:
    """
    Preprocess a point cloud by downsampling and computing FPFH features.

    Parameters:
    - pcd: The original point cloud.
    - voxel_size: The size of the voxel grid for downsampling.

    Returns:
    - The downsampled point cloud and its FPFH features.
    """
    pcd_down = pcd  # If you want to downsample, uncomment the following line
    # pcd_down = pcd.voxel_down_sample(voxel_size)

    # Estimate normals for the downsampled point cloud
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # Compute FPFH features for the downsampled point cloud
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size: float, 
                    source: o3d.geometry.PointCloud, 
                    target: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.PointCloud, 
                                               o3d.geometry.PointCloud, 
                                               o3d.pipelines.registration.Feature, 
                                               o3d.pipelines.registration.Feature]:
    """
    Prepare a pair of point clouds for registration algorithms by downsampling and computing FPFH features.

    Parameters:
    - voxel_size: The size of the voxel grid for downsampling.
    - pcd0, pcd1: The two point clouds to be prepared.

    Returns:
    - Tuple of the original and downsampled source and target point clouds and their features.
    """

    # # Apply an initial transformation to the source point cloud
    # trans_init = np.eye(4)
    # source.transform(trans_init)

    # Preprocess the source and target point clouds
    source_down, source_features = preprocess_point_cloud(source, voxel_size)
    target_down, target_features = preprocess_point_cloud(target, voxel_size)

    return source_down, target_down, source_features, target_features


def ICP(source: o3d.geometry.PointCloud, 
        target: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Apply the Iterative Closest Point (ICP) algorithm to a pair of point clouds.

    Parameters:
    - source, target: The source and target point clouds to be registered.

    Returns:
    - The ICP registration result.
    """
    threshold = 0.1  # Distance threshold for point-to-point correspondence
    trans_init = np.identity(4)  # Initial transformation

    # Apply ICP registration
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))

    estimated_transform = result_icp.transformation

    return estimated_transform


def FGR(source: o3d.geometry.PointCloud, 
        target: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Perform Fast Global Registration (FGR) on a pair of point clouds.

    Parameters:
    - source, target: Numpy arrays representing point clouds to be registered.

    Returns:
    - Registration result from FGR algorithm.
    """
    # Define voxel size for point cloud preprocessing
    voxel_size = 0.05

    # Prepare point clouds for FGR
    source_down, target_down, source_features, target_features = prepare_dataset(voxel_size, source, target)

    distance_threshold = voxel_size * 3  # Distance threshold based on voxel size

    # Perform fast global registration
    result_fgr = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_features, target_features,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))

    estimated_transform = result_fgr.transformation
    return estimated_transform


def RANSAC(source: o3d.geometry.PointCloud, 
           target: o3d.geometry.PointCloud) ->  np.ndarray:
    """
    Perform RANSAC-based registration on a pair of point clouds.

    Parameters:
    source, target: Point clouds to be registered as numpy arrays.

    Returns:
    Registration result from RANSAC algorithm.
    """
    # Define voxel size for point cloud preprocessing
    voxel_size = 0.05

    # Prepare point clouds for RANSAC
    source_down, target_down, source_features, target_features = prepare_dataset(voxel_size, source, target)

    # Define distance threshold based on voxel size
    distance_threshold = voxel_size * 1.5

    # Perform global registration using RANSAC
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_features, target_features, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    estimated_transform = result_ransac.transformation

    return estimated_transform


def multi_scale_ICP(source: o3d.geometry.PointCloud, 
                    target: o3d.geometry.PointCloud, 
                    voxel_sizes: List[float] = [0.1, 0.05, 0.01], 
                    max_iteration: List[int] = [50, 30, 14]) -> np.ndarray:
    """
    Perform a coarse-to-fine (multi-resolution) ICP on a pair of point clouds.

    Parameters:
    pc0, pc1: Point clouds to be registered.
    voxel_sizes: List of voxel sizes used for downsampling at each scale.
    max_iteration: List of maximum number of iterations at each scale.

    Returns:
    Registration result from the ICP algorithm.
    """
    assert len(voxel_sizes) == len(max_iteration), "The voxel sizes and maximum iteration arrays must be of the same length"
    
    trans_init = np.identity(4)  # Initial Transformation

    # Apply ICP at each scale
    for voxel_size, max_iter in zip(voxel_sizes, max_iteration):
        source_down, _ = preprocess_point_cloud(source, voxel_size)
        target_down, _ = preprocess_point_cloud(target, voxel_size)

        threshold = voxel_size * 1.5

        result_icp = o3d.pipelines.registration.registration_icp(
            source_down, target_down, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
        
        trans_init = result_icp.transformation

    estimated_transform = result_icp.transformation

    return estimated_transform


def point_to_plane_ICP(source: o3d.geometry.PointCloud, 
                       target: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Perform point-to-plane ICP on a pair of point clouds.

    Parameters:
    source, target: Point clouds to be registered.

    Returns:
    Registration result from the point-to-plane ICP algorithm.
    """
    
    voxel_size = 0.05  # Define voxel size for point cloud preprocessing

    # Compute surface normals for downsampled point clouds
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=30))

    threshold = 0.1  # Define distance thresh
    
    trans_init = np.identity(4)  # Define initial transformation

    # Apply Point-to-Plane ICP registration
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))

    estimated_transform = result_icp.transformation

    return estimated_transform


