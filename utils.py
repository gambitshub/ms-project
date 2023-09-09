"""
Utils.py

Author: James Daniels

This file contains a collection of utility functions used in the processing and evaluation 
of 3D point cloud data.
"""
import open3d as o3d
import os
import numpy as np
from pathlib import Path
import copy
import pandas as pd
import torch
from typing import List


def load_point_cloud(filepath: str) -> o3d.geometry.PointCloud:
    """
    Loads a point cloud file and converts it to an Open3D point cloud.

    Parameters:
    - filepath: The path to the point cloud file.

    Returns:
    - An Open3D point cloud.
    """
    # Create a Path object for the file path
    filepath = Path(filepath)

    # Check if the file is a csv file
    if filepath.suffix == '.csv':
        # Read the file into a pandas DataFrame
        df = pd.read_csv(filepath)

        # Convert the DataFrame to an Open3D point cloud
        pcd = o3d.geometry.PointCloud()

        # Set the points of the point cloud to the values of the DataFrame
        pcd.points = o3d.utility.Vector3dVector(df[['x', 'y', 'z']].values)
    else:
        # If the file is not a csv, read the point cloud file directly
        pcd = o3d.io.read_point_cloud(str(filepath))

    return pcd


def calculate_total_points(pcd1: o3d.geometry.PointCloud, 
                 pcd2: o3d.geometry.PointCloud) -> int:
    """
    Compute the total number of points in two point clouds.

    Parameters:
    pcd1 (o3d.geometry.PointCloud): The first point cloud.
    pcd2 (o3d.geometry.PointCloud): The second point cloud.

    Returns:
    int: The total number of points in the two point clouds.
    """
    return len(pcd1.points) + len(pcd2.points)


def overlap_percentage(pcd1: o3d.geometry.PointCloud, 
                       pcd2: o3d.geometry.PointCloud) -> float:
    """
    Compute the overlap percentage between two point clouds.

    Parameters:
    pcd1 (o3d.geometry.PointCloud): The first point cloud.
    pcd2 (o3d.geometry.PointCloud): The second point cloud.
    gt (np.ndarray): The ground truth transformation matrix.

    Returns:
    float: The overlap percentage between the two point clouds.
    """
    # Convert the points of the point clouds to numpy arrays
    cloud1 = np.asarray(pcd1.points)
    cloud2 = np.asarray(pcd2.points)

    # Compute the axis-aligned bounding boxes of each point cloud
    min1, max1 = np.min(cloud1, axis=0), np.max(cloud1, axis=0)
    min2, max2 = np.min(cloud2, axis=0), np.max(cloud2, axis=0)

    # Compute the intersection of the two bounding boxes 
    intersection_min = np.maximum(min1, min2)
    intersection_max = np.minimum(max1, max2)

    # Check if there is an intersection between the bounding boxes. If the intersection minimum is greater than the intersection
    # maximum in any dimension, there is no intersection.
    if np.any(intersection_min > intersection_max):
        return 0.0

    # Compute the volume of the intersection
    intersection_volume = np.prod(intersection_max - intersection_min)

    # Compute the volume of the bounding box of the first point cloud
    volume1 = np.prod(max1 - min1)

    # The overlap percentage is the ratio of the intersection volume to the volume of the first point cloud
    return intersection_volume / volume1


# def count_ply_files(directory: str) -> int:
#     """
#     Count the number of ply files in a directory.

#     Parameters:
#     directory (str): Path to the directory.

#     Returns:
#     int: Number of ply files in the directory.
#     """
#     # Count the number of files in the directory that have a .ply extension
#     return len([f for f in os.listdir(directory) if f.endswith('.ply')])


# def count_pcd_files(directory: str) -> int:
#     """
#     Count the number of pcd files in a directory.

#     Parameters:
#     directory (str): Path to the directory.

#     Returns:
#     int: Number of pcd files in the directory.
#     """
#     # Count the number of files in the directory that have a .pcd extension
#     return len([f for f in os.listdir(directory) if f.endswith('.pcd')])


def save_array_to_txt(array: np.ndarray, 
                      file_name: str) -> None:
    """
    Saves a given array as a text file with the specified file name.

    Parameters:
    array (np.ndarray): The array to be saved as a text file.
    file_name (str): The name of the text file.
    """
    # Open the file
    with open(file_name, 'w') as file:
        # For each item in the array, convert it to a string and write it to the file
        for item in array:
            line = ' '.join(map(str, item))
            file.write(f"{line}\n")


def save_transformations(T: np.ndarray, 
                         T_gt: np.ndarray, 
                         algorithm_name: str, 
                         dataset_name: str, 
                         index: int) -> None:
    """
    Saves the estimated and ground truth transformations to a text file.

    Parameters:
    T (np.ndarray): Estimated transformation matrix.
    T_gt (np.ndarray): Ground truth transformation matrix.
    algorithm_name (str): Name of the algorithm used.
    dataset_name (str): Name of the dataset.
    index (int): Index of the point cloud pair.
    """
    # Create a directory to save the results
    results_folder = Path(f"results/{dataset_name}/{algorithm_name}/transformations")
    results_folder.mkdir(parents=True, exist_ok=True)
    result_file_path = results_folder / f"{index}.txt"

    # Ensure T is a numpy array if it's a torch tensor
    if isinstance(T, torch.Tensor):
        T = T.cpu().numpy()

    # Open the file
    with open(result_file_path, 'w') as file:
        # Write the estimated transformation to the file
        file.write("Estimated Transformation:\n")
        file.write('\n'.join(' '.join(map(str, row)) for row in T))
        
        # Write the ground truth transformation to the file
        file.write("\n\nGround Truth Transformation:\n")
        file.write('\n'.join(' '.join(map(str, row)) for row in T_gt))


def make_open3d_point_cloud(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Creates an open3d point cloud object from given xyz coordinates.

    Parameters:
    xyz (np.ndarray): XYZ coordinates for the point cloud.

    Returns:
    o3d.geometry.PointCloud: The created point cloud object.
    """
    if isinstance(xyz, o3d.geometry.PointCloud):
        return xyz
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        return pcd


def draw_registration_result(source:  o3d.geometry.PointCloud, 
                             target:  o3d.geometry.PointCloud, 
                             transformation: np.ndarray, 
                             name: str = "Result") -> None:
    """
    Function to visualize the result of registration.
    Draws the registration results between source and target point clouds using Open3D.
    Draws the registration result of source and target point clouds after applying the transformation.
    

    Parameters:
    source (np.ndarray): The source point cloud data.
    target (np.ndarray): The target point cloud data.
    transformation (np.ndarray): The transformation matrix to apply on the source point cloud.
    name (str): The window name for the visualization. Defaults to "Result".
    """
    # Deepcopy is used to prevent any changes to the original data
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    # source_temp.paint_uniform_color([1, 1, 1])
    # target_temp.paint_uniform_color([1, 1, 1])

    # Painting each point cloud with a different color for visual distinction
    source_temp.paint_uniform_color([1, 0.706, 0])  # Gold color for source
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Blue color for target

    # Applying the estimated transformation to the source point cloud
    source_temp.transform(transformation)
    
    # Visualizing the transformed source and original target point clouds
    o3d.visualization.draw_geometries([source_temp, target_temp],width=1200,height=800,left=400,top=150,window_name=name)

