import os
import numpy as np
import open3d as o3d
from pathlib import Path

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
    volume = calculate_bounding_box_volume(pcd)
    num_points = calculate_number_of_points(pcd)

    # Calculate and return point density
    return num_points / volume if volume > 0 else 0

def average_bounding_box_volume_and_number_of_points(dataset_name: str):
    """
    Calculates the average bounding box volume and average number of points 
    for all point clouds in a SUN3D dataset.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset, used to construct the path to the directory containing 
        the point cloud files.

    Returns
    -------
    None.
    """

    # Initialize lists to hold the bounding box volumes and number of points for all files
    all_volumes = []
    all_points = []
    all_densities = []

    # Construct the path to the directory containing the point cloud files
    directory = Path(".") / "data/SUN3D" / dataset_name

    # Iterate over the specific files in the directory
    for i in range(38):  # Assuming there won't be more files than a very large number
        file_name = f"cloud_bin_{i}.ply"
        file_path = directory / file_name
        # print(f"Trying to open file: {file_path}")  # Debug print statement
        if file_path.is_file():
            # print(f"Processing file: {file_path}")  # Debug print statement
            # Read the point cloud file
            pcd = o3d.io.read_point_cloud(str(file_path))

            # Calculate the bounding box volume and number of points
            volume = calculate_bounding_box_volume(pcd)
            num_points = calculate_number_of_points(pcd)
            density = calculate_point_density(pcd)

            # Add the volume, number of points and density to the total for this file
            all_volumes.append(volume)
            all_points.append(num_points)
            all_densities.append(density)
            print(f"Volume, points, and density calculated for file: {file_path}")  # Debug print statement
        else:
            print(f"No such file: {file_path}")  # Debug print statement
            break

    # Calculate the average bounding box volume, number of points, and density for each file
    average_volume = np.median(all_volumes)
    average_points = np.median(all_points)
    average_density = np.median(all_densities)

    print(f'Average bounding box volume for {dataset_name}: {average_volume}')
    print(f'Average number of points for {dataset_name}: {average_points}')
    print(f'Average point density for {dataset_name}: {average_density}')


# Example usage:
average_bounding_box_volume_and_number_of_points("sun3d-hotel_umd-maryland_hotel3")
