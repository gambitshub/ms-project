"""
evaluation_metrics.py

Author: James Daniels

"""
import numpy as np
import copy
import math
from utils import make_open3d_point_cloud, draw_registration_result, calculate_total_points
from pathlib import Path
import os
import open3d as o3d
import json
from transformation import extract_rotation_translation, extract_parameters_from_transformation, extract_rotation_parameters


def get_full_evaluation_metrics(pcd0: np.ndarray, 
                                pcd1: np.ndarray, 
                                ground_truth_transform: np.ndarray, 
                                estimated_transform: np.ndarray, 
                                runtime: float,
                                downsampled_points: int) -> dict:
    """
    Computes evaluation metrics for a pair of point clouds.

    Parameters:
    pcd0 (np.ndarray): The first point cloud data.
    pcd1 (np.ndarray): The second point cloud data.
    ground_truth_transform (np.ndarray): The ground truth transformation matrix.
    estimated_transform (np.ndarray): The estimated transformation matrix.

    Returns:
    dict: The calculated metrics.
    """

    # Check if any of the transformation matrices are None, and if so, skip calculation
    if estimated_transform is None or ground_truth_transform is None:
        print("Either the estimated or ground truth transformation is None. Skipping metric calculation.")
        return None

    # Calculate Relative Translation Error (RTE)
    rte = get_rte(estimated_transform, ground_truth_transform)

    # Calculate Relative Rotation Error (RRE)
    rre = get_rre(estimated_transform, ground_truth_transform)

    # Create an open3D point cloud from the source data
    source = make_open3d_point_cloud(pcd0)

    # Calculate Root Mean Square Error (RMSE)
    rmse = get_rsme(source, estimated_transform, ground_truth_transform)

    # Calculate Mean Absolute Error (MAE)
    mae = get_mae(source, estimated_transform, ground_truth_transform)

    # Calculate combined error metric
    error_metric = calculate_error(source, estimated_transform, ground_truth_transform)

    # centroid, _ = pcd0.compute_mean_and_covariance()
    # point_t_err = point_translation_error(estimated_transform, ground_truth_transform, centroid)
    # print(point_t_err)

    # Compile all metrics into a dictionary
    metrics = {
        'runtime': runtime,
        'rmse': rmse,
        'rte': rte,
        'rre': rre,
        'mae': mae,
        'combined_error_metric' : error_metric,
        'downsampled_points' : downsampled_points
    }
    # print(metrics)

    return metrics


def save_full_eval_metrics(metrics_list: list, 
                           metrics_file_path: str) -> None:
    """
    Save the evaluation metrics to a .txt file.

    Parameters:
    metrics_list (list): The list of evaluation metrics to be saved.
    metrics_file_path (str): Path of the metrics file.
    """
    # Make sure the directory exists before opening the file
    os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)

    with open(metrics_file_path, 'w') as f:
        # Write each metric dictionary as a line in JSON format
        for metrics_dict in metrics_list:
            json.dump(metrics_dict, f)
            f.write("\n")
    print(f"Metrics saved to {metrics_file_path}")


def get_rre(T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    if T_pred is None:
        return False, np.inf, np.inf
    
    eps = float=1e-16
        # Compute the relative rotation error (RRE)
    rre = np.arccos(
        np.clip((np.trace(T_pred[:3, :3].T @ T_gt[:3, :3]) - 1) / 2, -1 + eps,
                1 - eps)) * 180 / math.pi
    print(rre)
    
    # rotation_pred, t0 = extract_rotation_translation(T_pred)
    # _, angle_pred = extract_rotation_parameters(rotation_pred)
    # print('\n angle pred = ')
    # print(angle_pred)
    # print(t0)

    # rotation_gt, t1 = extract_rotation_translation(T_gt)
    # axis, angle_gt = extract_rotation_parameters(rotation_gt)
    # print('\n angle gt = ')
    # print(angle_gt)
    # print(t1)
    # print(axis)

    # rotation_error = abs(angle_pred - angle_gt)
    # print(rotation_error)
    # rotation_error_degrees = np.degrees(rotation_error)
    # print(rotation_error_degrees)

    # # Compute the relative rotation error (RRE)
    # rre = rotation_error_degrees
    # # print(rre)

    return rre


def get_rte(T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    # Compute the relative translation error (RTE)
    rte = np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3])

    return rte

def point_translation_error(T_pred: np.ndarray, T_gt: np.ndarray, point: np.ndarray) -> float:
    """
    Compute the translation error for a specific point.

    Parameters:
    - T_pred (np.ndarray): The predicted transformation matrix.
    - T_gt (np.ndarray): The ground truth transformation matrix.
    - point (np.ndarray): The point to be transformed.

    Returns:
    - float: The translation error for the point.
    """
    # Extend the point to homogeneous coordinates
    point_hom = np.append(point, 1)

    # Apply the transformations to the point
    transformed_point_pred = np.matmul(T_pred, point_hom)
    transformed_point_gt = np.matmul(T_gt, point_hom)

    # Compute the Euclidean distance between the transformed points
    error = np.linalg.norm(transformed_point_pred[:3] - transformed_point_gt[:3])

    return error

def get_rsme(pcd: np.ndarray, 
             T_pred: np.ndarray, 
             T_gt: np.ndarray) -> float:
    """
    Compute the RSME between the point cloud transformed by the estimated and ground truth transformation matrices.

    Parameters:
    pcd (np.ndarray): The point cloud data.
    T_pred (np.ndarray): The predicted transformation matrix.
    T_gt (np.ndarray): The ground truth transformation matrix.

    Returns:
    float: The RMSE between the point cloud after transformations.
    """
    # Apply the estimated transformation to the point cloud
    pcd_est = copy.deepcopy(pcd)
    pcd_est.transform(T_pred)
    
    # Apply the ground truth transformation to the point cloud
    pcd_true = copy.deepcopy(pcd)
    pcd_true.transform(T_gt)

    # Convert the point clouds to numpy arrays
    pcd_est = np.asarray(pcd_est.points)
    pcd_true = np.asarray(pcd_true.points)

    # Compute the RMSE
    return np.sqrt(np.mean((pcd_est - pcd_true) ** 2))


def get_mae(pcd: np.ndarray, 
            T_pred: np.ndarray, 
            T_gt: np.ndarray) -> float:
    """
    Compute the Mean Absolute Error (MAE) between a point cloud transformed by the estimated and ground truth transformations.

    Parameters:
    pcd (np.ndarray): The point cloud data.
    T_pred (np.ndarray): The predicted transformation matrix.
    T_gt (np.ndarray): The ground truth transformation matrix.

    Returns:
    float: The MAE.
    """
    pcd_est = copy.deepcopy(pcd)
    pcd_est.transform(T_pred)
    pcd_true = copy.deepcopy(pcd)
    pcd_true.transform(T_gt)

    pcd_est = np.asarray(pcd_est.points)
    pcd_true = np.asarray(pcd_true.points)

    return np.mean(np.abs(pcd_est - pcd_true))

def calculate_error(pcd: o3d.geometry.PointCloud, 
                    T_pred: np.ndarray, 
                    T_gt: np.ndarray) -> float:
    """
    Calculate a combined error metric based on transformations and point cloud data.

    Parameters:
    pcd (o3d.geometry.PointCloud): The point cloud data.
    T_pred (np.ndarray): The predicted transformation matrix.
    T_gt (np.ndarray): The ground truth transformation matrix.

    Returns:
    float: The combined error metric.
    """
    # Apply the estimated transformation to the point cloud
    cloud1 = copy.deepcopy(pcd)
    cloud1.transform(T_pred)
    
    # Apply the ground truth transformation to the point cloud
    cloud2 = copy.deepcopy(pcd)
    cloud2.transform(T_gt)
    
    centroid, _ = cloud1.compute_mean_and_covariance()
    weights = np.linalg.norm(np.asarray(cloud1.points) - centroid, 2, axis=1)
    distances = np.linalg.norm(np.asarray(cloud1.points) - np.asarray(cloud2.points), 2, axis=1)/len(weights)
    return np.sum(distances/weights)


def read_metrics_file(file_path):
    """Read a metrics file and return the processed metrics."""
    # Check if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            # Read the metrics file line by line, convert each line to a list of floats
            metrics = [[float(x) for x in line.strip().split()] for line in file.readlines()]
        return metrics
    else:
        return None

