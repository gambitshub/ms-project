import os
import numpy as np
import open3d as o3d
from pathlib import Path
import json
import pandas as pd
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns


def load_point_clouds_from_folder(dataset_name: str):
    """
    Load all point clouds in the given dataset folder.

    Parameters:
    dataset_name (str): Name of the dataset.

    Returns:
    List of point cloud objects.
    """
    # Folder where the point clouds are stored
    point_cloud_folder = Path(f"{dataset_name}/experiment_point_clouds")

    # Load all .pcd files in the folder
    point_clouds = [o3d.io.read_point_cloud(str(point_cloud_folder / file_name))
                    for file_name in os.listdir(point_cloud_folder)
                    if file_name.endswith('.pcd')]

    return point_clouds


def calculate_average_points(point_clouds):
    """
    Calculates the average number of points in a list of point clouds.

    Parameters:
    point_clouds (List[PointCloud]): The point clouds.

    Returns:
    float: The average number of points.
    """
    # Count total points and total files
    total_points = sum(len(pcd.points) for pcd in point_clouds)
    total_files = len(point_clouds)

    # Calculate and return average number of points
    return total_points / total_files if total_files > 0 else 0


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


def calculate_nearest_neighbor_distances(pcd):
    """
    Calculates the mean nearest neighbor distance for a point cloud.
    """
    # Convert to numpy array
    pcd_np = np.asarray(pcd.points)

    # Calculate distances to nearest neighbor (excluding self)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pcd_np)
    distances, _ = nbrs.kneighbors(pcd_np)

    # Return mean distance
    return np.mean(distances[:, 1])

def calculate_bounding_box_size(pcd):
    """
    Calculates the bounding box size in a point cloud.
    """
    return np.max(np.asarray(pcd.points), axis=0) - np.min(np.asarray(pcd.points), axis=0)

def calculate_center_point(pcd):
    """
    Calculates the center point in a point cloud.
    """
    return (np.max(np.asarray(pcd.points), axis=0) + np.min(np.asarray(pcd.points), axis=0)) / 2

def calculate_surface_normals(pcd):
    """
    Calculates the mean normal vector for a point cloud.
    """
    pcd.estimate_normals()
    return np.mean(np.asarray(pcd.normals), axis=0)

def calculate_noise(pcd):
    """
    Calculates the variance in nearest neighbor distances for a point cloud.
    """
    pcd_np = np.asarray(pcd.points)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pcd_np)
    distances, _ = nbrs.kneighbors(pcd_np)
    return np.var(distances[:, 1])

def calculate_outliers(pcd):
    """
    Calculates the percentage of points that are considered outliers in a point cloud.
    """
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    outlier_count = len(pcd.points) - len(ind)
    return outlier_count / len(pcd.points)

def calculate_uniformity(pcd):
    """
    Calculates the uniformity of point distributions in a point cloud.
    """
    pcd_np = np.asarray(pcd.points)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pcd_np)
    distances, _ = nbrs.kneighbors(pcd_np)
    return np.var(distances[:, 1])



def calculate_properties(dataset_names):
    """
    Calculates point cloud properties for all given dataset names and saves them in the respective folders.

    Parameters:
    dataset_names (List[str]): List of dataset names.
    """
    # Define properties
    properties = {
        "Number of points": lambda pcd: len(pcd.points),
        "Point density": calculate_point_density,
        "Bounding box size": lambda pcd: np.max(pcd.points, axis=0) - np.min(pcd.points, axis=0),
        "Center point": lambda pcd: (np.max(pcd.points, axis=0) + np.min(pcd.points, axis=0)) / 2,
        "Nearest neighbor distance": calculate_nearest_neighbor_distances,
        "Normal vector": calculate_surface_normals,
        "Noise": calculate_noise,
        "% Outliers": calculate_outliers,
        "Uniformity": calculate_uniformity
    }

    # Calculate and save properties for each dataset
    for dataset_name in dataset_names:
        print(dataset_name)
        properties_file = f"{dataset_name}/experiment_point_clouds/properties.txt"
        
        # If properties file exists, skip this dataset
        if os.path.isfile(properties_file):
            print(f"Properties file for {dataset_name} already exists. Skipping calculation.")
            continue

        # Initialize empty lists for each property
        calculated_properties = {name: [] for name in properties.keys()}

        # Load and process each point cloud individually
        point_cloud_folder = Path(f"{dataset_name}/experiment_point_clouds")
        for file_name in os.listdir(point_cloud_folder):
            if file_name.endswith('.pcd'):
                # Load point cloud
                pcd = o3d.io.read_point_cloud(str(point_cloud_folder / file_name))

                # Calculate and store properties
                for name, func in properties.items():
                    calculated_properties[name].append(func(pcd))

        # Calculate averages                                                                                                                                                                                                                                                                                                                                                                                                                                
        averaged_properties = {name: np.mean(values) for name, values in calculated_properties.items()}
                                                                                                                                                
        # Save properties as a JSON file
        with open(properties_file, 'w') as file:
            file.write(json.dumps(averaged_properties, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))
        

def read_and_display_properties(dataset_names):
    """
    Reads all properties files and displays them in a table.

    Parameters:
    dataset_names (List[str]): List of dataset names.
    """
    # List to store all properties
    all_properties = []

    # Read properties from each file
    for dataset_name in dataset_names:
        with open(f"{dataset_name}/experiment_point_clouds/properties.txt", 'r') as file:
            properties = json.load(file)

        # Add dataset name to properties
        properties["Dataset Name"] = dataset_name

        # Add properties to list
        all_properties.append(properties)

    # Create DataFrame
    df = pd.DataFrame(all_properties)

    # Set dataset names as index
    df.set_index("Dataset Name", inplace=True)

    # Display DataFrame
    print(df)


def visualize_dataset_properties(dataset_names):
    """
    Reads all properties files, displays them in a table and visualizes them.

    Parameters:
    dataset_names (List[str]): List of dataset names.
    """
    # List to store all properties
    all_properties = []

    # Read properties from each file
    for dataset_name in dataset_names:
        with open(f"{dataset_name}/experiment_point_clouds/properties.txt", 'r') as file:
            properties = json.load(file)

        # Add dataset name to properties
        properties["Dataset Name"] = dataset_name

        # Add properties to list
        all_properties.append(properties)

    # Create DataFrame
    df = pd.DataFrame(all_properties)

    # Set dataset names as index
    df.set_index("Dataset Name", inplace=True)

    # Display DataFrame
    # print(df)

    # Create bar plots for each property
    for column in df.columns:
        plt.figure(figsize=(10, 10))  # Adjust as needed
        sns.barplot(x=df.index, y=df[column])
        plt.title(column)
        plt.xticks(rotation=90)  # Rotate x-axis labels for readability
        plt.tight_layout()

        # Save the plot
        plt.savefig(f'graphresults/{column}.png')

        # Clear the plot so the next one doesn't overlap
        plt.clf()

