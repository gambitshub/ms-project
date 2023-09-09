import os
from pathlib import Path
import numpy as np
import open3d as o3d

def calculate_bounding_box_size(pcd):
    points = np.asarray(pcd.points)
    bounding_box_max = np.max(points, axis=0)
    bounding_box_min = np.min(points, axis=0)

    return bounding_box_max - bounding_box_min

def calculate_bounding_box_volume(pcd):
    size = calculate_bounding_box_size(pcd)
    # print(size)
    volume = size[0] * size[1] * size[2]
    
    return volume

def calculate_number_of_points(pcd):
    return len(pcd.points)

def calculate_point_density(pcd):
    volume = calculate_bounding_box_volume(pcd)
    num_points = calculate_number_of_points(pcd)

    return num_points / volume if volume > 0 else 0


def calculate_point_density_knn(pcd, k=5):
    """
    Calculates the point density in a point cloud using k-nearest neighbors.

    Parameters:
    pcd (PointCloud): The point cloud.
    k (int): The number of neighbors to consider.

    Returns:
    np.array: An array of densities for each point in the point cloud.
    """
    # Build a KDTree for efficient nearest neighbor search
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    densities = []
    for i in range(np.asarray(pcd.points).shape[0]):
        # Find the k nearest neighbors
        _, idx, _ = pcd_tree.search_knn_vector_3d(pcd.points[i], k)
        
        # Calculate the distance to the kth neighbor
        kth_distance = np.linalg.norm(np.asarray(pcd.points)[i] - np.asarray(pcd.points)[idx[-1]])

        # Calculate and store the density
        densities.append(k / kth_distance**3 if kth_distance > 0 else 0)

    return np.array(densities)


def average_metrics(directory_path: str):
    """
    Calculates the average bounding box volume, number of points and point density for all point clouds in a directory.

    Parameters
    ----------
    directory_path : str
        The path to the directory containing the point cloud files.

    Returns
    -------
    None.
    """
    # Initialize dictionaries to hold the total metrics for each file
    total_volumes = {}
    total_points = {}
    total_densities = {}
    new_densities = {}

    # Walk through the directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.ply'):
                # Read the point cloud file
                pcd = o3d.io.read_point_cloud(os.path.join(root, file))
                print(file)
                # Calculate the bounding box volume
                volume = calculate_bounding_box_volume(pcd)
                print(volume)
                # Calculate the number of points
                num_points = calculate_number_of_points(pcd)
                # Calculate the point density
                density = calculate_point_density(pcd)

                # new_density = calculate_point_density_knn(pcd)

                # Add the metrics to the totals for this file
                total_volumes.setdefault(file, []).append(volume)
                total_points.setdefault(file, []).append(num_points)
                total_densities.setdefault(file, []).append(density)
                # new_densities.setdefault(file, []).append(new_density)

    print(total_volumes['sfm.ply'])
    # Calculate the average metrics for each file
    for file in total_volumes.keys():
        average_volume = np.median(total_volumes[file])
        average_points = np.median(total_points[file])
        average_density = np.median(total_densities[file])
        # average_new_density = np.mean(new_densities[file])
        
        print(f'Average metrics for {file}:')
        print(f'Bounding Box Volume: {average_volume}')
        print(f'Number of Points: {average_points}')
        print(f'Point Density: {average_density}\n')
        # print(f'new density: {average_new_density}\n')


# Example usage:
average_metrics("./data/cross-source-dataset")
