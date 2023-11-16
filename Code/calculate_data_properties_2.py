import os
import numpy as np
import open3d as o3d
from pathlib import Path
import pandas as pd

def calculate_bounding_box_size(pcd):
    """
    Calculates the bounding box size in a point cloud.
    """
    points = np.asarray(pcd.points)
    bounding_box_max = np.max(points, axis=0)
    bounding_box_min = np.min(points, axis=0)

    return bounding_box_max - bounding_box_min

def calculate_bounding_box_volume(pcd):
    """
    Calculates the volume of the bounding box in a point cloud.
    """
    size = calculate_bounding_box_size(pcd)
    
    # Assumes that the size is a 3D vector. If not, this will fail.
    volume = size[0] * size[1] * size[2]
    
    return volume

def calculate_number_of_points(pcd):
    """
    Calculates the number of points in a point cloud.
    """
    return len(pcd.points)

def calculate_point_density(pcd):
    """
    Calculates the point density in a point cloud.

    Parameters:
    pcd (PointCloud): The point cloud.

    Returns:
    float: The point density.
    """
    # Calculate volume and number of points
    volume = (np.asarray(pcd.points).max(axis=0) - np.asarray(pcd.points).min(axis=0)).prod()
    num_points = len(pcd.points)

    # Calculate and return point density
    return num_points / volume if volume > 0 else 0

def average_bounding_box_volume_and_number_of_points(dataset_name: str):
    """
    Calculates the average bounding box volume, average number of points, and average point density 
    for all point clouds in an ETH dataset.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset, used to construct the path to the directory containing 
        the point cloud files.

    Returns
    -------
    None.
    """

    # Initialize lists to hold the bounding box volumes, number of points and point densities for all files
    all_volumes = []
    all_points = []
    all_densities = []

    # Construct the path to the directory containing the point cloud files
    directory = Path(".") / "data/ETH" / dataset_name

    # Get a list of all point cloud files in the directory
    filenames = [f for f in os.listdir(directory) if f.startswith('PointCloud') and f.endswith('.csv')]

    # Read each file and convert to an open3d point cloud
    for i, filename in enumerate(filenames):
        # Read the file into a pandas DataFrame
        df = pd.read_csv(os.path.join(directory, filename))

        # Convert the DataFrame to an open3d point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(df[['x', 'y', 'z']].values)

        # Calculate the bounding box volume, number of points and point density
        volume = calculate_bounding_box_volume(pcd)
        num_points = calculate_number_of_points(pcd)
        density = calculate_point_density(pcd)

        # Add the volume, number of points, and point density to the total for this file
        all_volumes.append(volume)
        all_points.append(num_points)
        all_densities.append(density)

    # Calculate the average bounding box volume, number of points and point density for each file
    average_volume = np.median(all_volumes)
    average_points = np.median(all_points)
    average_density = np.median(all_densities)

    print(f'Average bounding box volume for {dataset_name}: {average_volume}')
    print(f'Average number of points for {dataset_name}: {average_points}')
    print(f'Average point density for {dataset_name}: {average_density}')



# Example usage:
average_bounding_box_volume_and_number_of_points("gazebo_winter")
